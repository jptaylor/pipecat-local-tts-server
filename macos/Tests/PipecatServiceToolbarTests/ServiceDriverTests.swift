import Foundation
import Testing

@testable import PipecatToolbarCore

/// `ServiceDriver` running the lifecycle's effects through fakes: a server
/// process the test drives, an environment, and Apple Speech's install.
@MainActor @Suite struct ServiceDriverTests {
    /// A server process whose output and exit the test controls.
    final class FakeProcess: ServerProcess, @unchecked Sendable {
        let pid: Int32 = 4242
        let stdout: AsyncStream<String>
        let stderr: AsyncStream<String>
        private let out: AsyncStream<String>.Continuation
        private let err: AsyncStream<String>.Continuation
        private let lock = NSLock()
        private var exitWaiters: [CheckedContinuation<Int32, Never>] = []
        private var status: Int32?
        private(set) var stops = 0
        /// False: `stop()` only records the request; the test calls `exit`.
        var exitsOnStop = true

        init() {
            (stdout, out) = AsyncStream.makeStream()
            (stderr, err) = AsyncStream.makeStream()
        }

        func run() throws {}
        func emit(_ line: String) { out.yield(line) }

        func exit(_ code: Int32) {
            out.finish()
            err.finish()
            let waiters: [CheckedContinuation<Int32, Never>] = lock.withLock {
                status = code
                defer { exitWaiters = [] }
                return exitWaiters
            }
            for waiter in waiters { waiter.resume(returning: code) }
        }

        func stop(grace: Duration) async -> Int32 {
            lock.withLock { stops += 1 }
            if exitsOnStop { exit(15) }
            return 15
        }

        func waitForExit(after reader: Task<Void, Never>, drainLimit: Duration) async -> Int32 {
            let code = await withCheckedContinuation { (c: CheckedContinuation<Int32, Never>) in
                let done: Int32? = lock.withLock {
                    if let status { return status }
                    exitWaiters.append(c)
                    return nil
                }
                if let done { c.resume(returning: done) }
            }
            await reader.value
            return code
        }
    }

    struct FakeEnvironment: ServiceEnvironment {
        func environmentStatus(_ backend: Service) -> EnvironmentStatus {
            .ready(EnvironmentMarker(service: backend.rawValue, fingerprint: "w", python: "3.12", createdAt: Date()))
        }
        func isRequested(_ service: Service) -> Bool { true }
        func modelCacheStatus(_ backend: Service, includeOutdated: Bool) async -> ModelCacheStatus? { nil }
        func install(_ backend: Service, mode: InstallMode) -> AsyncThrowingStream<InstallEvent, Error> {
            AsyncThrowingStream { $0.finish() }
        }
        func remove(_ backend: Service, deleteModel: Bool, knownFolders: [URL]) async throws {}
        func serveArguments(for info: ServiceInfo, port: Int) -> [String]? { ["serve"] }
    }

    @MainActor final class FakeHost: ServiceHost {
        let serviceEnvironment: (any ServiceEnvironment)? = FakeEnvironment()
        let servicePaths: AppPaths?
        var tryIt: [String] = []

        init() {
            let root = FileManager.default.temporaryDirectory.appending(path: "driver-tests-\(UUID().uuidString)")
            servicePaths = AppPaths(supportDir: root, cacheDir: root.appending(path: "cache"), logsDir: root.appending(path: "logs"))
        }
        func familyChanged() async {}
        func openTryIt(port: Int) { tryIt.append("open \(port)") }
        func detachTryIt() { tryIt.append("detach") }
        func closeTryIt() { tryIt.append("close") }
    }

    @MainActor final class Flag { var value = false }

    /// Polls (on the main actor) until `condition` holds or 2 s pass.
    func until(_ condition: () -> Bool) async {
        let deadline = ContinuousClock.now + .seconds(2)
        while !condition(), ContinuousClock.now < deadline { try? await Task.sleep(for: .milliseconds(5)) }
    }

    func makeDriver(_ process: FakeProcess, host: FakeHost, configure: (inout ServiceDependencies) -> Void = { _ in })
        -> ServiceDriver
    {
        var deps = ServiceDependencies()
        deps.isPortFree = { _ in true }
        deps.makeProcess = { _ in process }
        configure(&deps)
        let driver = ServiceDriver(info: Service.tone.info, dependencies: deps)
        driver.host = host
        return driver
    }

    // MARK: - Ported from the app-target ServiceModelTests

    /// App quit cancels an Apple Speech model install and waits for it to unwind.
    @Test func shutdownCancelsAndAwaitsAnAppleSpeechInstall() async {
        var deps = ServiceDependencies()
        let ended = Flag()
        deps.appleSpeechPhase = { .notInstalled }
        deps.appleSpeechInstall = { _ in
            defer { ended.value = true }
            try await Task.sleep(for: .seconds(30))
        }
        let driver = ServiceDriver(info: Service.appleSpeech.info, dependencies: deps)
        driver.install()
        await Task.yield()
        let clock = ContinuousClock()
        let start = clock.now
        await driver.shutdown()
        #expect(ended.value, "shutdown returned with the install still running")
        #expect(clock.now - start < .seconds(2))
        #expect(driver.state.phase == .installing(.model), "a cancel during quit shows nothing")
    }

    /// An install that ignores cancellation holds quit for at most `installDrain`.
    @Test func shutdownWaitsForAStuckInstallOnlySoLong() async {
        var deps = ServiceDependencies()
        deps.installDrain = .milliseconds(200)
        deps.appleSpeechInstall = { _ in
            let deadline = ContinuousClock.now + .seconds(3)
            while ContinuousClock.now < deadline { try? await Task.sleep(for: .milliseconds(20)) }
        }
        let driver = ServiceDriver(info: Service.appleSpeech.info, dependencies: deps)
        driver.install()
        let clock = ContinuousClock()
        let start = clock.now
        await driver.shutdown()
        #expect(clock.now - start < .seconds(2))
    }

    // MARK: - Server lifecycle through the driver

    @Test func startReadyStop() async {
        let process = FakeProcess()
        let host = FakeHost()
        let driver = makeDriver(process, host: host)
        await driver.refresh()
        #expect(driver.state.phase == .installed)
        driver.start()
        #expect(driver.state.phase == .starting)
        #expect(driver.startDetail == "Loading model…")
        process.emit(#"{"event":"ready","port":8665,"rate":24000,"pid":77}"#)
        await until { driver.state.phase == .running }
        #expect(driver.state.running?.pid == 77)
        #expect(host.tryIt == ["open 8665"])
        driver.stop()
        #expect(driver.state.phase == .stopping)
        await until { driver.state.phase == .installed }
        #expect(driver.state.phase == .installed)
        #expect(host.tryIt == ["open 8665", "detach", "close"])
        #expect(driver.state.server == nil)
    }

    /// A ready line that lands after Stop must not flip the phase back to running.
    @Test func aReadyAfterStopIsIgnored() async {
        let process = FakeProcess()
        let host = FakeHost()
        process.exitsOnStop = false
        let driver = makeDriver(process, host: host)
        driver.start()
        driver.stop()
        #expect(driver.state.phase == .stopping)
        process.emit(#"{"event":"ready","port":8665}"#)
        try? await Task.sleep(for: .milliseconds(50))
        #expect(driver.state.phase == .stopping, "the late ready flipped the phase")
        process.exit(15)
        await until { driver.state.server == nil }
        #expect(driver.state.phase == .installed)
        #expect(!host.tryIt.contains("open 8665"))
    }

    /// The ready timeout stops a server that never reports ready and fails the start.
    @Test func aServerThatNeverGetsReadyTimesOut() async {
        let process = FakeProcess()
        let host = FakeHost()  // the driver holds its host weakly
        let driver = makeDriver(process, host: host) { $0.readyTimeout = .milliseconds(50) }
        driver.start()
        await until { driver.state.server == nil }
        #expect(process.stops == 1)
        guard case .failed(let failure) = driver.state.phase else {
            Issue.record("\(driver.state.phase)")
            return
        }
        #expect(failure.kind == .startup(code: "Timeout"))
    }

    /// An `error` line is judged with the exit that follows it.
    @Test func aSupervisedErrorFailsTheStart() async {
        let process = FakeProcess()
        let host = FakeHost()
        let driver = makeDriver(process, host: host)
        driver.start()
        process.emit(#"{"event":"error","code":"ConfigError","message":"bad voice"}"#)
        process.exit(1)
        await until { driver.state.server == nil }
        #expect(driver.state.phase == .failed(ServiceFailure(kind: .startup(code: "ConfigError"), title: "Couldn't start", message: "bad voice")))
    }
}

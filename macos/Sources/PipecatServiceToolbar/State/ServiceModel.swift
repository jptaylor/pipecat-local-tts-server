import AppKit
import Foundation
import Observation
import PipecatToolbarCore

/// UI state and lifecycle for one service: install, start/stop, remove.
///
/// Python services (TTS server, STT helper) run as supervised child processes;
/// Apple Speech runs in-process behind `STTWebSocketServer`. Everything here is
/// main-actor state; the work happens off it and reports back through streams.
@MainActor
@Observable
final class ServiceModel: Identifiable {
    enum Phase: Equatable {
        case checking
        /// Can't run on this Mac (OS too old, language unsupported).
        case unavailable(String)
        case notInstalled
        /// The environment is installed but the model weights are missing (e.g. a cancelled download).
        case needsModel
        case installing(InstallStage)
        case installed
        case starting
        case running
        case stopping
        case removing
        case failed(Failure)
    }

    struct Failure: Equatable {
        enum Kind: Equatable {
            case install, download, portBusy(PortProbe.Owner), startup(code: String?), crashed(status: Int32)
        }
        var kind: Kind
        var title: String
        var message: String
        var logTail: [String] = []

        /// Failures of a server the user started (shown with the running services).
        var isRuntime: Bool {
            switch kind {
            case .portBusy, .startup, .crashed: true
            case .install, .download: false
            }
        }
    }

    struct RunningInfo: Equatable {
        var pid: Int32
        var port: Int
        var rate: Int
        var model: String?
        var since: Date
    }

    let info: ServiceInfo
    nonisolated var id: Service { info.service }
    var service: Service { info.service }
    var kind: ServiceKind { info.kind }

    private(set) var phase: Phase = .checking
    private(set) var download: DownloadProgress?
    /// Apple Speech model install progress (0...1).
    private(set) var assetProgress: Double?
    private(set) var installDetail: String = ""
    private(set) var environment: EnvironmentStatus = .missing
    /// Installed by the user (a variant's family environment can exist without it).
    private(set) var requested = false
    private(set) var cache: ModelCacheStatus?
    private(set) var running: RunningInfo?
    private(set) var startDetail: String = ""
    private(set) var log: [String] = []
    private(set) var warning: String?
    var approxDownloadBytes: Int64?
    private(set) var ttsTryIt: TryItSession?
    private(set) var sttTryIt: STTTryItSession?

    @ObservationIgnored weak var app: AppModel?
    @ObservationIgnored private var installTask: Task<Void, Never>?
    /// Internal (not private) for the state-machine tests.
    @ObservationIgnored var server: Subprocess?
    @ObservationIgnored private var nativeServer: STTWebSocketServer?
    @ObservationIgnored private var logSink: LogSink?
    @ObservationIgnored var stopRequested = false
    /// Downloads the Apple Speech model (replaced by the tests).
    @ObservationIgnored var appleSpeechInstall: (@escaping @MainActor (Double) -> Void) async throws -> Void = AppleSpeech.install
    /// How long app quit waits for a cancelled Apple Speech install to unwind.
    @ObservationIgnored var installDrain: Duration = .seconds(5)
    @ObservationIgnored private var shuttingDown = false
    @ObservationIgnored private var supervisedError: (code: String?, message: String?)?

    init(info: ServiceInfo) {
        self.info = info
    }

    // MARK: - Derived

    var isRunning: Bool { phase == .running }

    /// Started by the user: shown in the main menu (running, on its way up or
    /// down, or failed at runtime).
    var isActive: Bool {
        switch phase {
        case .starting, .running, .stopping: true
        case .failed(let failure): failure.isRuntime
        default: false
        }
    }

    var isInstalled: Bool {
        switch phase {
        case .installed, .needsModel, .starting, .running, .stopping: true
        case .unavailable, .notInstalled, .checking: false
        case .installing, .removing: requested
        case .failed(let failure): failure.isRuntime || requested
        }
    }

    var isInstalling: Bool {
        if case .installing = phase { return true }
        return false
    }

    var isBusy: Bool {
        switch phase {
        case .checking, .installing, .starting, .stopping, .removing: true
        default: false
        }
    }

    /// The Python services own an environment and model files that can be deleted.
    var canDeleteModel: Bool { info.runtime.usesPython && isInstalled && phase != .removing }

    var installFraction: Double? { info.runtime == .appleSpeech ? assetProgress : download?.fraction }
    var diskBytes: Int64? { cache.map(\.sizeBytes).flatMap { $0 > 0 ? $0 : nil } }
    var logURL: URL? { app?.paths.appLog(service) }
    var port: Int { running?.port ?? info.port }

    // MARK: - Refresh

    func refresh() async {
        if info.runtime == .appleSpeech {
            switch await AppleSpeech.status() {
            case .installed: phase = .installed
            case .needsDownload: phase = .notInstalled
            case .downloading: phase = .installing(.model)
            case .unavailable(let why): phase = .unavailable(why)
            }
            return
        }
        guard let manager = app?.manager else { return }
        environment = manager.environmentStatus(service)
        requested = environment != .missing && manager.isRequested(service)
        guard requested else {
            phase = .notInstalled
            cache = nil
            return
        }
        switch environment {
        case .missing:
            phase = .notInstalled
            cache = nil
        case .outdated:
            // Packages are reinstalled before the next start; assume weights are fine.
            phase = .installed
        case .ready:
            let status = await manager.modelCacheStatus(service)
            cache = status
            phase = (status?.cached ?? true) ? .installed : .needsModel
        }
    }

    /// After a sibling variant changed the shared environment: re-read it, but
    /// leave a phase that work in progress (or a failure on screen) owns.
    func environmentChanged() async {
        switch phase {
        case .checking, .notInstalled, .needsModel, .installed: await refresh()
        default: environment = app?.manager?.environmentStatus(service) ?? environment
        }
    }

    // MARK: - Install

    func install() {
        guard installTask == nil else { return }
        if info.runtime == .appleSpeech { return installAppleSpeech() }
        guard let manager = app?.manager else { return }
        let mode: InstallMode
        switch (environment, phase) {
        case (.missing, _): mode = .full
        case (_, .needsModel): mode = .model
        case (_, .failed(let f)) where f.kind == .download: mode = .model
        // The manager skips the steps a sibling's environment already covers.
        default: mode = .full
        }
        runInstall(manager: manager, mode: mode) { [weak self] in
            await self?.refresh()
        }
    }

    func cancelInstall() {
        installTask?.cancel()
    }

    private func installAppleSpeech() {
        warning = nil
        assetProgress = 0
        phase = .installing(.model)
        Log.install.info("apple_speech: installing speech model")
        let install = appleSpeechInstall
        installTask = Task { [weak self] in
            do {
                try await install { fraction in self?.assetProgress = fraction }
                self?.installTask = nil
                self?.assetProgress = nil
                await self?.refresh()
            } catch {
                guard let self else { return }
                self.installTask = nil
                self.assetProgress = nil
                // Quit cancelled it: the app is going away, nothing to show.
                if self.shuttingDown { return }
                if Task.isCancelled { return await self.refresh() }
                self.phase = .failed(Failure(kind: .download, title: "Model download failed", message: error.localizedDescription))
            }
        }
    }

    private func runInstall(manager: EnvironmentManager, mode: InstallMode, then: @escaping @MainActor () async -> Void) {
        warning = nil
        download = nil
        installDetail = ""
        switch (mode, environment) {
        case (.model, _), (.full, .ready): phase = .installing(.model)
        case (.packages, _), (.full, .outdated): phase = .installing(.packages)
        case (.full, .missing): phase = .installing(.python)
        }
        let service = self.service
        Log.install.info("\(service.rawValue, privacy: .public): install (\(String(describing: mode), privacy: .public))")
        installTask = Task { [weak self] in
            var stage: InstallStage = .python
            do {
                // `consume` throws on Cancel; a plain `for try await` would end
                // normally and run `then` (which restarts a start()'s install).
                try await manager.install(service, mode: mode).consume { event in
                    guard let self else { return }
                    switch event {
                    case .stage(let s):
                        stage = s
                        self.phase = .installing(s)
                        self.installDetail = ""
                    case .log(let line):
                        self.installDetail = Self.summarize(uvLine: line) ?? self.installDetail
                        self.appendLog(line)
                    case .download(let progress):
                        self.download = progress
                    case .warning(let message):
                        self.warning = message
                    }
                }
                guard let self else { return }
                self.installTask = nil
                self.download = nil
                Log.install.info("\(service.rawValue, privacy: .public): install finished")
                await then()
                await self.app?.familyChanged(by: self)
            } catch {
                guard let self else { return }
                self.installTask = nil
                self.download = nil
                // remove() cancelled this install and owns the phase from here.
                if self.phase == .removing { return }
                if error is CancellationError || Task.isCancelled {
                    await self.refresh()
                    await self.app?.familyChanged(by: self)
                    return
                }
                Log.install.error("\(service.rawValue, privacy: .public): install failed: \(error.localizedDescription, privacy: .public)")
                let installError = error as? InstallError
                let isDownload = (installError?.stage ?? stage) == .model
                self.phase = .failed(
                    Failure(
                        kind: isDownload ? .download : .install,
                        title: isDownload ? "Model download failed" : "Install failed",
                        message: Self.friendly(error.localizedDescription),
                        logTail: installError?.logTail ?? []))
            }
        }
    }

    /// Short progress text from uv's output.
    static func summarize(uvLine line: String) -> String? {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        for prefix in ["Resolved", "Prepared", "Installed", "Downloading", "Uninstalled", "Using CPython", "Building"]
        where trimmed.hasPrefix(prefix) {
            return trimmed
        }
        return nil
    }

    static func friendly(_ message: String) -> String {
        if message.contains("Failed to fetch") || message.contains("dns error") || message.contains("ConnectError")
            || message.contains("Network is unreachable")
        {
            return "Couldn't reach the network. Check your connection and try again."
        }
        return message
    }

    // MARK: - Serve

    func start() {
        guard let app, server == nil, nativeServer == nil, phase != .removing else { return }
        if info.runtime.usesPython, case .outdated = environment, let manager = app.manager {
            // The app was updated since this env was built: refresh its packages, then start.
            runInstall(manager: manager, mode: .packages) { [weak self] in
                await self?.refresh()
                if !Task.isCancelled { self?.start() }
            }
            return
        }
        let port = info.port
        guard PortProbe.isFree(port) else {
            phase = .starting
            startDetail = "Checking port \(port)…"
            Task {
                let owner = await PortProbe.owner(of: port, backend: service)
                self.phase = .failed(Self.portBusyFailure(port: port, owner: owner))
            }
            return
        }
        if info.runtime == .appleSpeech {
            launchAppleSpeech(app: app, port: port)
        } else {
            launch(app: app, port: port)
        }
    }

    private static func portBusyFailure(port: Int, owner: PortProbe.Owner) -> Failure {
        var message = "Port \(port) is already in use by \(owner.summary)."
        if let label = owner.launchdLabel {
            message += " A launchd agent (\(label)) is loaded; stop it before starting here."
        }
        return Failure(kind: .portBusy(owner), title: "Port \(port) is busy", message: message)
    }

    private func becameReady(port: Int, pid: Int32, rate: Int, model: String?) {
        running = RunningInfo(pid: pid, port: port, rate: rate, model: model, since: Date())
        phase = .running
        switch kind {
        case .tts: ttsTryIt = TryItSession(backend: service, port: port)
        case .stt: sttTryIt = STTTryItSession(port: port)
        }
        Log.server.info("\(self.service.rawValue, privacy: .public): ready on port \(port)")
    }

    private func launchAppleSpeech(app: AppModel, port: Int) {
        let sink = LogSink(url: app.paths.appLog(service))
        sink.write("---- \(Date().formatted(.iso8601)) starting apple_speech on \(ServiceCatalog.host):\(port)")
        logSink = sink
        stopRequested = false
        startDetail = "Starting…"
        phase = .starting
        let server = STTWebSocketServer(identity: .init(name: "apple_speech", model: "SpeechTranscriber")) { pcm, language in
            try await AppleSpeech.transcribe(pcm16: pcm, language: language)
        }
        server.onLog = { [weak self] line in
            Task { @MainActor in
                self?.logSink?.write(line)
                self?.appendLog(line)
            }
        }
        nativeServer = server
        Task {
            do {
                let bound = try await server.start(port: port)
                // stop() (or remove) while binding: that server is gone, don't report it.
                guard self.nativeServer === server, self.phase == .starting else { return server.stop() }
                self.appendLog("listening on \(ServiceCatalog.host):\(bound)")
                self.becameReady(port: bound, pid: ProcessInfo.processInfo.processIdentifier, rate: 16000, model: "SpeechTranscriber")
            } catch {
                guard self.nativeServer === server else { return }
                self.nativeServer = nil
                self.phase = .failed(Failure(kind: .startup(code: nil), title: "Couldn't start", message: error.localizedDescription))
            }
        }
    }

    private func launch(app: AppModel, port: Int) {
        guard let manager = app.manager, let args = manager.serveArguments(for: info, port: port) else {
            phase = .failed(Failure(kind: .startup(code: nil), title: "Couldn't start", message: "This build is missing the STT helper."))
            return
        }
        let sub = Subprocess(
            .init(executable: app.paths.python(service), arguments: args, environment: ProcessEnvironment.huggingFace(), holdStdin: true))
        let sink = LogSink(url: app.paths.appLog(service))
        sink.write("---- \(Date().formatted(.iso8601)) starting \(service.rawValue) on \(ServiceCatalog.host):\(port)")
        do {
            try sub.run()
        } catch {
            sink.close()
            phase = .failed(Failure(kind: .startup(code: nil), title: "Couldn't start", message: error.localizedDescription))
            return
        }
        server = sub
        logSink = sink
        Log.server.info("\(self.service.rawValue, privacy: .public): started pid \(sub.pid) on port \(port)")
        stopRequested = false
        supervisedError = nil
        startDetail = "Loading model…"
        phase = .starting

        Task { [weak self] in
            for await line in sub.stderr {
                guard let self else { return }
                self.logSink?.write(line)
                self.appendLog(line)
                if self.phase == .starting, let message = Self.logMessage(line) { self.startDetail = message }
            }
        }
        let events = Task { [weak self] in
            for await line in sub.stdout {
                guard let self, let event = ServerCLIEvent.parse(line) else { continue }
                // A ready that lands after Stop/Delete must not flip the phase back to running.
                if event.event == "ready", self.server === sub, self.phase == .starting {
                    self.becameReady(
                        port: event.port ?? port, pid: event.pid ?? sub.pid,
                        rate: event.rate ?? (self.kind == .tts ? 24000 : 16000), model: event.model)
                } else if event.event == "error" {
                    self.supervisedError = (event.code, event.message)
                }
            }
        }
        Task { [weak self] in
            // The final `error` line is read before the exit is judged.
            let status = await sub.waitForExit(after: events)
            self?.serverExited(sub, status: status)
        }
        Task { [weak self] in
            try? await Task.sleep(for: .seconds(600))
            guard let self, self.server === sub, self.phase == .starting else { return }
            self.supervisedError = ("Timeout", "The server did not become ready within 10 minutes.")
            _ = await sub.stop()
        }
    }

    /// The server process `sub` exited. A requested exit only completes a Stop
    /// (`.stopping` -> `.installed`): it is reported after the stdout drain, up
    /// to 2 s after `shutdown()` returned, so a removal (or its failure) may own
    /// the phase by then and must keep it.
    func serverExited(_ sub: Subprocess, status: Int32) {
        guard server === sub else { return }
        Log.server.info("\(self.service.rawValue, privacy: .public): exited status \(status) (requested: \(self.stopRequested))")
        server = nil
        tearDownSession(note: "exited with status \(status)")
        if stopRequested {
            if phase == .stopping { phase = .installed }
            return
        }
        let tail = Array(log.suffix(8))
        if let err = supervisedError {
            if err.code == SupervisedErrorCode.addressInUse.rawValue {
                Task {
                    let owner = await PortProbe.owner(of: info.port, backend: service)
                    self.phase = .failed(Self.portBusyFailure(port: info.port, owner: owner))
                }
                return
            }
            let title = err.code == SupervisedErrorCode.moduleNotFound.rawValue ? "Environment incomplete" : "Couldn't start"
            phase = .failed(Failure(kind: .startup(code: err.code), title: title, message: err.message ?? "Startup failed.", logTail: tail))
        } else {
            phase = .failed(
                Failure(
                    kind: .crashed(status: status), title: "Server stopped unexpectedly",
                    message: "The \(info.displayName) server exited with status \(status).", logTail: tail))
        }
    }

    private func tearDownSession(note: String) {
        running = nil
        ttsTryIt?.detach()
        ttsTryIt = nil
        sttTryIt?.cancel()
        sttTryIt = nil
        logSink?.write("---- \(note)")
        logSink?.close()
        logSink = nil
    }

    func stop() {
        if let native = nativeServer {
            native.stop()
            nativeServer = nil
            tearDownSession(note: "stopped")
            phase = .installed
            return
        }
        guard let sub = server else { return }
        stopRequested = true
        phase = .stopping
        ttsTryIt?.detach()
        Task { _ = await sub.stop(grace: .seconds(10)) }
    }

    /// For app quit (and before removal): stop and wait. Quit also cancels an
    /// Apple Speech model install and waits (at most `installDrain`) for it to
    /// unwind; a Python install is the environment manager's to stop.
    func shutdown() async {
        if info.runtime == .appleSpeech, let install = installTask {
            shuttingDown = true
            install.cancel()
            await Self.wait(for: install, atMost: installDrain)
        }
        if nativeServer != nil {
            stop()
            return
        }
        guard let sub = server else { return }
        stopRequested = true
        ttsTryIt?.detach()
        _ = await sub.stop(grace: .seconds(10))
    }

    /// Returns when `task` ends or `limit` passed, whichever is first (a task
    /// that ignores cancellation can't hold app quit).
    private static func wait(for task: Task<Void, Never>, atMost limit: Duration) async {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            var resumed = false
            let resume: @MainActor () -> Void = {
                guard !resumed else { return }
                resumed = true
                continuation.resume()
            }
            Task { @MainActor in
                await task.value
                resume()
            }
            Task { @MainActor in
                try? await Task.sleep(for: limit)
                resume()
            }
        }
    }

    func dismissFailure() {
        Task { await refresh() }
    }

    // MARK: - Remove

    /// Uninstalls the service and, with `deleteModel`, the model files no installed
    /// sibling shares. The family environment goes with its last installed variant.
    func remove(deleteModel: Bool) {
        guard info.runtime.usesPython, let manager = app?.manager, phase != .removing else { return }
        phase = .removing
        // An install in flight is cancelled and must unwind (and drop the family
        // lock) before removal; it leaves the phase to this task.
        let install = installTask
        install?.cancel()
        Task {
            await install?.value
            self.phase = .removing  // an install that had already finished may have refreshed it
            // Stop the server first so its exit can't race the removal.
            await shutdown()
            let known = (cache?.repoFolders ?? []) + (cache?.derivedFolders ?? [])
            do {
                try await manager.remove(service, deleteModel: deleteModel, knownFolders: known)
            } catch {
                self.phase = .failed(Failure(kind: .install, title: "Couldn't remove", message: error.localizedDescription))
                return
            }
            Log.install.info("\(self.service.rawValue, privacy: .public): removed (model files: \(deleteModel))")
            self.log = []
            await self.refresh()
            await self.app?.familyChanged(by: self)
        }
    }

    // MARK: - Logs

    private func appendLog(_ line: String) {
        log.append(line)
        if log.count > 300 { log.removeFirst(log.count - 300) }
    }

    /// `2026-10-23 09:31:34,650 INFO tts_server: message` -> `message`
    static func logMessage(_ line: String) -> String? {
        let parts = line.split(separator: " ", maxSplits: 3, omittingEmptySubsequences: false)
        guard parts.count == 4, let colon = parts[3].range(of: ": ") else {
            return line.count < 120 && !line.contains("|") ? line : nil
        }
        var message = String(parts[3][colon.upperBound...])
        if message.hasPrefix("tts_server: ") { message.removeFirst("tts_server: ".count) }
        return message
    }

    func revealLog() {
        guard let url = logURL, FileManager.default.fileExists(atPath: url.path) else { return }
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }
}

#if DEBUG
    extension ServiceModel {
        /// Puts the model into a given state for `--snapshot` rendering (debug builds only).
        func setPreviewState(
            phase: Phase, download: DownloadProgress? = nil, running: RunningInfo? = nil,
            environment: EnvironmentStatus = .missing, requested: Bool? = nil, diskBytes: Int64? = nil,
            log: [String] = [], startDetail: String = "", installDetail: String = ""
        ) {
            self.phase = phase
            self.download = download
            self.running = running
            self.environment = environment
            self.requested = requested ?? (environment != .missing)
            self.cache = diskBytes.map { ModelCacheStatus(cached: true, sizeBytes: $0, repoFolders: []) }
            self.log = log
            self.startDetail = startDetail
            self.installDetail = installDetail
            if let running {
                switch kind {
                case .tts: ttsTryIt = TryItSession(backend: service, port: running.port)
                case .stt: sttTryIt = STTTryItSession(port: running.port)
                }
            }
        }
    }
#endif

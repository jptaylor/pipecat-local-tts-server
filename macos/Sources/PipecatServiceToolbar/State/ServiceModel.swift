import AppKit
import Foundation
import Observation
import PipecatToolbarCore

/// UI state for one service: publishes its `ServiceDriver` (Core's lifecycle
/// state machine and effect runner) and forwards the user's intents to it.
///
/// It adds only what needs the app target: the Try it sessions, Apple Speech
/// (Speech framework), the derived flags the views read, and Finder.
@MainActor
@Observable
final class ServiceModel: Identifiable {
    typealias Phase = ServicePhase
    typealias Failure = ServiceFailure
    typealias RunningInfo = ServiceRunningInfo

    let info: ServiceInfo
    let driver: ServiceDriver
    nonisolated var id: Service { info.service }
    var service: Service { info.service }
    var kind: ServiceKind { info.kind }

    var phase: Phase { driver.state.phase }
    var download: DownloadProgress? { driver.download }
    /// Apple Speech model install progress (0...1).
    var assetProgress: Double? { driver.assetProgress }
    var installDetail: String { driver.installDetail }
    var environment: EnvironmentStatus { driver.state.environment }
    /// Installed by the user (a variant's family environment can exist without it).
    var requested: Bool { driver.state.requested }
    var cache: ModelCacheStatus? { driver.cache }
    var running: RunningInfo? { driver.state.running }
    var startDetail: String { driver.startDetail }
    var log: [String] { driver.log }
    var warning: String? { driver.warning }
    var approxDownloadBytes: Int64?
    private(set) var ttsTryIt: TryItSession?
    private(set) var sttTryIt: STTTryItSession?

    @ObservationIgnored weak var app: AppModel?

    init(info: ServiceInfo) {
        self.info = info
        var dependencies = ServiceDependencies()
        dependencies.appleSpeechPhase = {
            switch await AppleSpeech.status() {
            case .installed: .installed
            case .needsDownload: .notInstalled
            case .downloading: .installing(.model)
            case .unavailable(let why): .unavailable(why)
            }
        }
        dependencies.appleSpeechInstall = { progress in try await AppleSpeech.install(progress: progress) }
        dependencies.makeNativeServer = {
            STTWebSocketServer(identity: .init(name: "apple_speech", model: "SpeechTranscriber")) { pcm, language in
                try await AppleSpeech.transcribe(pcm16: pcm, language: language)
            }
        }
        driver = ServiceDriver(info: info, dependencies: dependencies)
        driver.host = self
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

    // MARK: - Intents

    func refresh() async { await driver.refresh() }
    /// After a sibling variant changed the shared environment: re-read it, but
    /// leave a phase that work in progress (or a failure on screen) owns.
    func environmentChanged() async { await driver.environmentChanged() }
    func install() { driver.install() }
    func cancelInstall() { driver.cancelInstall() }
    func start() { driver.start() }
    func stop() { driver.stop() }
    /// For app quit (and before removal): stop and wait.
    func shutdown() async { await driver.shutdown() }
    func dismissFailure() { driver.dismissFailure() }
    func remove(deleteModel: Bool) { driver.remove(deleteModel: deleteModel) }

    static func logMessage(_ line: String) -> String? { ServiceDriver.logMessage(line) }

    func revealLog() {
        guard let url = logURL, FileManager.default.fileExists(atPath: url.path) else { return }
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }
}

extension ServiceModel: ServiceHost {
    var serviceEnvironment: (any ServiceEnvironment)? { app?.manager }
    var servicePaths: AppPaths? { app?.paths }

    func familyChanged() async { await app?.familyChanged(by: self) }

    func openTryIt(port: Int) {
        switch kind {
        case .tts: ttsTryIt = TryItSession(backend: service, port: port)
        case .stt: sttTryIt = STTTryItSession(port: port)
        }
    }

    func detachTryIt() { ttsTryIt?.detach() }

    func closeTryIt() {
        ttsTryIt?.detach()
        ttsTryIt = nil
        sttTryIt?.cancel()
        sttTryIt = nil
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
            var state = LifecycleState(info: info)
            state.phase = phase
            state.running = running
            state.environment = environment
            state.requested = requested ?? (environment != .missing)
            driver.setPreviewState(
                state, download: download, cache: diskBytes.map { ModelCacheStatus(cached: true, sizeBytes: $0, repoFolders: []) },
                log: log, startDetail: startDetail, installDetail: installDetail)
            if let running { openTryIt(port: running.port) }
        }
    }
#endif

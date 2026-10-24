import Foundation

// The per-service lifecycle as a pure state machine: `ServiceLifecycle.reduce`
// maps (state, event) to (state', effects). It decides every phase change and
// every guard against a late event (a ready or exit that arrives after Stop,
// Delete or a relaunch). `ServiceDriver` runs the effects; the app's
// `ServiceModel` only publishes the driver's state and forwards user intents.

/// Where one service is in its lifecycle (what the UI shows).
public enum ServicePhase: Equatable, Sendable {
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
    case failed(ServiceFailure)
}

public struct ServiceFailure: Equatable, Sendable {
    public enum Kind: Equatable, Sendable {
        case install, download, portBusy(PortProbe.Owner), startup(code: String?), crashed(status: Int32)
    }
    public var kind: Kind
    public var title: String
    public var message: String
    public var logTail: [String]

    public init(kind: Kind, title: String, message: String, logTail: [String] = []) {
        self.kind = kind
        self.title = title
        self.message = message
        self.logTail = logTail
    }

    /// Failures of a server the user started (shown with the running services).
    public var isRuntime: Bool {
        switch kind {
        case .portBusy, .startup, .crashed: true
        case .install, .download: false
        }
    }

    static func portBusy(port: Int, owner: PortProbe.Owner) -> ServiceFailure {
        var message = "Port \(port) is already in use by \(owner.summary)."
        if let label = owner.launchdLabel {
            message += " A launchd agent (\(label)) is loaded; stop it before starting here."
        }
        return ServiceFailure(kind: .portBusy(owner), title: "Port \(port) is busy", message: message)
    }
}

public struct ServiceRunningInfo: Equatable, Sendable {
    public var pid: Int32
    public var port: Int
    public var rate: Int
    public var model: String?
    public var since: Date

    public init(pid: Int32, port: Int, rate: Int, model: String?, since: Date) {
        self.pid = pid
        self.port = port
        self.rate = rate
        self.model = model
        self.since = since
    }
}

/// One launched server (a subprocess or the in-process Apple Speech server).
/// Events name the server they come from, so one from a replaced server is ignored.
public struct ServerID: Hashable, Sendable, CustomStringConvertible {
    public let raw: Int
    public init(_ raw: Int) { self.raw = raw }
    public var description: String { "#\(raw)" }
}

/// A supervised server's `{"event":"error"}` line (or the ready timeout).
public struct SupervisedError: Equatable, Sendable {
    public var code: String?
    public var message: String?
    public init(code: String?, message: String?) {
        self.code = code
        self.message = message
    }
}

/// What a finished install does next.
public enum InstallContinuation: Equatable, Sendable {
    case refresh
    /// The package refresh `start()` runs for an outdated environment: start once it is in.
    case refreshThenStart
}

public enum InstallOutcome: Equatable, Sendable {
    case finished
    case cancelled
    /// `stage` is the step that failed (`.model` reads as a download failure).
    case failed(stage: InstallStage, message: String, logTail: [String])
}

public enum LifecycleEvent: Equatable, Sendable {
    /// A re-read of the environment (nil: leave that field as it is).
    case refreshed(ServicePhase, environment: EnvironmentStatus? = nil, requested: Bool? = nil)
    /// A sibling variant changed the shared environment (nil: no manager yet).
    case environmentChanged(EnvironmentStatus?)
    case installRequested(managerAvailable: Bool)
    case cancelInstallRequested
    case installStage(InstallStage)
    case installEnded(InstallOutcome, then: InstallContinuation)
    case appleSpeechInstallEnded(InstallOutcome)
    case startRequested(appAvailable: Bool, managerAvailable: Bool, portFree: Bool)
    case portOwnerFound(PortProbe.Owner)
    case launched(ServerID)
    case launchFailed(String)
    case nativeLaunched(ServerID)
    case nativeBound(ServerID, port: Int, pid: Int32, at: Date)
    case nativeFailed(ServerID, String)
    case ready(ServerID, port: Int, pid: Int32, rate: Int, model: String?, at: Date)
    case supervisedError(ServerID, SupervisedError)
    case readyTimedOut(ServerID)
    case exited(ServerID, status: Int32, logTail: [String])
    case stopRequested
    /// App quit, first step: cancel an Apple Speech install.
    case shutdownRequested
    /// App quit (and removal), second step: stop the server.
    case shutdownServers
    case removeRequested(deleteModel: Bool, managerAvailable: Bool)
    /// The removal's cancelled install has unwound.
    case removalResumed(deleteModel: Bool)
    case removalFailed(String)
    case removalFinished
    case dismissFailure
}

/// Work for `ServiceDriver`, in order. Effects never change the phase themselves;
/// their results come back as events.
public enum LifecycleEffect: Equatable, Sendable {
    case refresh
    case familyChanged
    case runInstall(InstallMode, then: InstallContinuation)
    case runAppleSpeechInstall
    /// Cancels the install in flight (remembered for `awaitInstall`).
    case cancelInstall
    /// Waits for the cancelled install to unwind; `bounded` caps it at the quit drain.
    case awaitInstall(bounded: Bool)
    case probePortOwner(port: Int)
    case launch(port: Int)
    case launchNative(port: Int)
    /// Stops a native server that bound after Stop or a relaunch.
    case discardNative(ServerID)
    case stopNative
    case startReadyTimer(ServerID)
    case openTryIt(port: Int)
    case detachTryIt
    /// Ends the running session: Try it, the log file.
    case tearDown(note: String)
    /// SIGTERM, then SIGKILL after the stop grace; `wait` awaits the exit.
    case stopServer(ServerID, wait: Bool)
    case appendLog(String)
    case startDetail(String)
    case resumeRemoval(deleteModel: Bool)
    case shutdown
    case removeEnvironment(deleteModel: Bool)
    case clearLog
    /// `start()` again, unless the task running it was cancelled.
    case startIfNotCancelled
}

public struct LifecycleState: Equatable, Sendable {
    public let runtime: ServiceRuntime
    public let port: Int
    public let displayName: String

    public var phase: ServicePhase = .checking
    public var environment: EnvironmentStatus = .missing
    /// Installed by the user (a variant's family environment can exist without it).
    public var requested = false
    public var running: ServiceRunningInfo?
    /// The current server subprocess.
    public var server: ServerID?
    /// The current in-process (Apple Speech) server.
    public var native: ServerID?
    /// Stop (or quit, or removal) asked the current server to exit.
    public var stopRequested = false
    /// App quit began.
    public var shuttingDown = false
    /// An install is in flight.
    public var installing = false
    public var supervisedError: SupervisedError?

    public init(info: ServiceInfo) {
        runtime = info.runtime
        port = info.port
        displayName = info.displayName
    }
}

public enum ServiceLifecycle {
    /// The install a user's Install runs, from the environment and the phase.
    public static func installMode(environment: EnvironmentStatus, phase: ServicePhase) -> InstallMode {
        switch (environment, phase) {
        case (.missing, _): .full
        case (_, .needsModel): .model
        case (_, .failed(let f)) where f.kind == .download: .model
        // The manager skips the steps a sibling's environment already covers.
        default: .full
        }
    }

    /// The stage an install shows before its first progress event.
    public static func firstStage(mode: InstallMode, environment: EnvironmentStatus) -> InstallStage {
        switch (mode, environment) {
        case (.model, _), (.full, .ready): .model
        case (.packages, _), (.full, .outdated): .packages
        case (.full, .missing): .python
        }
    }

    /// The phase a refresh reads from disk (Python services).
    public static func installedPhase(environment: EnvironmentStatus, requested: Bool, cached: Bool?) -> ServicePhase {
        guard requested else { return .notInstalled }
        switch environment {
        case .missing: return .notInstalled
        // Packages are reinstalled before the next start; assume weights are fine.
        case .outdated: return .installed
        case .ready: return (cached ?? true) ? .installed : .needsModel
        }
    }

    public static func reduce(_ s: inout LifecycleState, _ event: LifecycleEvent) -> [LifecycleEffect] {
        switch event {
        case .refreshed(let phase, let environment, let requested):
            s.phase = phase
            if let environment { s.environment = environment }
            if let requested { s.requested = requested }
            return []

        case .environmentChanged(let environment):
            // Leave a phase that work in progress (or a failure on screen) owns.
            switch s.phase {
            case .checking, .notInstalled, .needsModel, .installed: return [.refresh]
            default:
                if let environment { s.environment = environment }
                return []
            }

        case .installRequested(let managerAvailable):
            guard !s.installing else { return [] }
            if s.runtime == .appleSpeech {
                s.installing = true
                s.phase = .installing(.model)
                return [.runAppleSpeechInstall]
            }
            guard managerAvailable else { return [] }
            return beginInstall(&s, mode: installMode(environment: s.environment, phase: s.phase), then: .refresh)

        case .cancelInstallRequested:
            return [.cancelInstall]

        case .installStage(let stage):
            s.phase = .installing(stage)
            return []

        case .installEnded(let outcome, let then):
            s.installing = false
            switch outcome {
            case .finished:
                return then == .refresh ? [.refresh, .familyChanged] : [.refresh, .startIfNotCancelled, .familyChanged]
            case _ where s.phase == .removing:
                // remove() cancelled this install and owns the phase from here.
                return []
            case .cancelled:
                return [.refresh, .familyChanged]
            case .failed(let stage, let message, let logTail):
                let download = stage == .model
                s.phase = .failed(
                    ServiceFailure(
                        kind: download ? .download : .install, title: download ? "Model download failed" : "Install failed",
                        message: message, logTail: logTail))
                return []
            }

        case .appleSpeechInstallEnded(let outcome):
            s.installing = false
            switch outcome {
            case .finished: return [.refresh]
            // Quit cancelled it: the app is going away, nothing to show.
            case _ where s.shuttingDown: return []
            case .cancelled: return [.refresh]
            case .failed(_, let message, _):
                s.phase = .failed(ServiceFailure(kind: .download, title: "Model download failed", message: message))
                return []
            }

        case .startRequested(let appAvailable, let managerAvailable, let portFree):
            guard appAvailable, s.server == nil, s.native == nil, s.phase != .removing else { return [] }
            if s.runtime.usesPython, case .outdated = s.environment, managerAvailable {
                // The app was updated since this env was built: refresh its packages, then start.
                return beginInstall(&s, mode: .packages, then: .refreshThenStart)
            }
            guard portFree else {
                s.phase = .starting
                return [.startDetail("Checking port \(s.port)…"), .probePortOwner(port: s.port)]
            }
            return s.runtime == .appleSpeech ? [.launchNative(port: s.port)] : [.launch(port: s.port)]

        case .portOwnerFound(let owner):
            s.phase = .failed(.portBusy(port: s.port, owner: owner))
            return []

        case .launched(let id):
            s.server = id
            s.stopRequested = false
            s.supervisedError = nil
            s.phase = .starting
            return [.startDetail("Loading model…"), .startReadyTimer(id)]

        case .launchFailed(let message):
            s.phase = .failed(ServiceFailure(kind: .startup(code: nil), title: "Couldn't start", message: message))
            return []

        case .nativeLaunched(let id):
            s.native = id
            s.stopRequested = false
            s.phase = .starting
            return [.startDetail("Starting…")]

        case .nativeBound(let id, let port, let pid, let date):
            // stop() (or remove) while binding: that server is gone, don't report it.
            guard s.native == id, s.phase == .starting else { return [.discardNative(id)] }
            return [.appendLog("listening on \(ServiceCatalog.host):\(port)")]
                + becameReady(&s, port: port, pid: pid, rate: STTWire.rate, model: "SpeechTranscriber", at: date)

        case .nativeFailed(let id, let message):
            guard s.native == id else { return [] }
            s.native = nil
            s.phase = .failed(ServiceFailure(kind: .startup(code: nil), title: "Couldn't start", message: message))
            return []

        case .ready(let id, let port, let pid, let rate, let model, let date):
            // A ready that lands after Stop/Delete must not flip the phase back to running.
            guard s.server == id, s.phase == .starting else { return [] }
            return becameReady(&s, port: port, pid: pid, rate: rate, model: model, at: date)

        case .supervisedError(_, let error):
            s.supervisedError = error
            return []

        case .readyTimedOut(let id):
            guard s.server == id, s.phase == .starting else { return [] }
            s.supervisedError = SupervisedError(code: "Timeout", message: "The server did not become ready within 10 minutes.")
            return [.stopServer(id, wait: false)]

        case .exited(let id, let status, let logTail):
            // A requested exit only completes a Stop (`.stopping` -> `.installed`):
            // it is reported after the stdout drain, up to 2 s after `shutdown()`
            // returned, so a removal (or its failure) may own the phase by then.
            guard s.server == id else { return [] }
            s.server = nil
            s.running = nil
            let effects: [LifecycleEffect] = [.tearDown(note: "exited with status \(status)")]
            if s.stopRequested {
                if s.phase == .stopping { s.phase = .installed }
                return effects
            }
            guard let error = s.supervisedError else {
                s.phase = .failed(
                    ServiceFailure(
                        kind: .crashed(status: status), title: "Server stopped unexpectedly",
                        message: "The \(s.displayName) server exited with status \(status).", logTail: logTail))
                return effects
            }
            if error.code == SupervisedErrorCode.addressInUse.rawValue {
                return effects + [.probePortOwner(port: s.port)]
            }
            let title = error.code == SupervisedErrorCode.moduleNotFound.rawValue ? "Environment incomplete" : "Couldn't start"
            s.phase = .failed(
                ServiceFailure(
                    kind: .startup(code: error.code), title: title, message: error.message ?? "Startup failed.",
                    logTail: logTail))
            return effects

        case .stopRequested:
            if s.native != nil { return stopNative(&s) }
            guard let id = s.server else { return [] }
            s.stopRequested = true
            s.phase = .stopping
            return [.detachTryIt, .stopServer(id, wait: false)]

        case .shutdownRequested:
            // Quit cancels an Apple Speech model install; a Python install is the
            // environment manager's to stop.
            guard s.runtime == .appleSpeech, s.installing else { return [] }
            s.shuttingDown = true
            return [.cancelInstall, .awaitInstall(bounded: true)]

        case .shutdownServers:
            if s.native != nil { return stopNative(&s) }
            guard let id = s.server else { return [] }
            s.stopRequested = true
            return [.detachTryIt, .stopServer(id, wait: true)]

        case .removeRequested(let deleteModel, let managerAvailable):
            guard s.runtime.usesPython, managerAvailable, s.phase != .removing else { return [] }
            s.phase = .removing
            // An install in flight is cancelled and must unwind (and drop the family
            // lock) before removal; it leaves the phase to the removal.
            return [.cancelInstall, .awaitInstall(bounded: false), .resumeRemoval(deleteModel: deleteModel)]

        case .removalResumed(let deleteModel):
            s.phase = .removing  // an install that had already finished may have refreshed it
            // Stop the server first so its exit can't race the removal.
            return [.shutdown, .removeEnvironment(deleteModel: deleteModel)]

        case .removalFailed(let message):
            s.phase = .failed(ServiceFailure(kind: .install, title: "Couldn't remove", message: message))
            return []

        case .removalFinished:
            return [.clearLog, .refresh, .familyChanged]

        case .dismissFailure:
            return [.refresh]
        }
    }

    private static func beginInstall(_ s: inout LifecycleState, mode: InstallMode, then: InstallContinuation)
        -> [LifecycleEffect]
    {
        s.installing = true
        s.phase = .installing(firstStage(mode: mode, environment: s.environment))
        return [.runInstall(mode, then: then)]
    }

    private static func becameReady(
        _ s: inout LifecycleState, port: Int, pid: Int32, rate: Int, model: String?, at date: Date
    ) -> [LifecycleEffect] {
        s.running = ServiceRunningInfo(pid: pid, port: port, rate: rate, model: model, since: date)
        s.phase = .running
        return [.openTryIt(port: port)]
    }

    private static func stopNative(_ s: inout LifecycleState) -> [LifecycleEffect] {
        s.native = nil
        s.running = nil
        s.phase = .installed
        return [.stopNative, .tearDown(note: "stopped")]
    }
}

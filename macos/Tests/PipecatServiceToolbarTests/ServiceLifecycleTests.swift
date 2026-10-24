import Foundation
import Testing

@testable import PipecatToolbarCore

/// `ServiceLifecycle.reduce`, the pure state machine behind every service:
/// late ready/exit/timeout events, Stop, Delete and quit against every phase.
@Suite struct ServiceLifecycleTests {
    static let failure = ServiceFailure(kind: .install, title: "Couldn't remove", message: "no")
    static let phases: [ServicePhase] = [
        .checking, .unavailable("old"), .notInstalled, .needsModel, .installing(.model), .installed,
        .starting, .running, .stopping, .removing, .failed(failure),
    ]
    static let current = ServerID(1)
    static let stale = ServerID(0)
    static let date = Date(timeIntervalSince1970: 0)
    static let marker = EnvironmentMarker(service: "tone", fingerprint: "w", python: "3.12", createdAt: date)

    static func state(
        _ phase: ServicePhase, info: ServiceInfo = Service.tone.info, server: ServerID? = current,
        stopRequested: Bool = false, shuttingDown: Bool = false, installing: Bool = false
    ) -> LifecycleState {
        var s = LifecycleState(info: info)
        s.phase = phase
        s.environment = .ready(marker)
        s.requested = true
        s.server = server
        s.stopRequested = stopRequested
        s.shuttingDown = shuttingDown
        s.installing = installing
        return s
    }

    // MARK: - Ported from the app-target ServiceModelTests

    /// {phase} x {requested exit, crash} x {the exiting server is current, or
    /// already replaced}. A requested exit only completes a Stop; in particular
    /// one reported after a removal finished (`.notInstalled`) or failed keeps
    /// that phase. A crash of the current server fails it; a replaced server's
    /// exit changes nothing.
    @Test func everyExitAgainstEveryPhase() {
        for phase in Self.phases {
            for requested in [true, false] {
                for current in [true, false] {
                    var s = Self.state(phase, server: current ? Self.current : ServerID(2), stopRequested: requested)
                    _ = ServiceLifecycle.reduce(&s, .exited(Self.current, status: 0, logTail: []))
                    let label = "\(phase) requested=\(requested) current=\(current)"
                    switch (current, requested) {
                    case (false, _): #expect(s.phase == phase, "\(label)")
                    case (true, true): #expect(s.phase == (phase == .stopping ? .installed : phase), "\(label)")
                    case (true, false):
                        guard case .failed(let f) = s.phase, case .crashed(status: 0) = f.kind else {
                            Issue.record("\(label): \(s.phase)")
                            continue
                        }
                    }
                    #expect(current ? s.server == nil : s.server != nil, "\(label)")
                }
            }
        }
    }

    /// The reported order: remove() -> shutdown() returns -> removal and
    /// refresh finish (`.notInstalled`) -> the drained exit arrives.
    @Test func aLateExitDoesNotReinstallARemovedService() {
        var s = Self.state(.installed)
        _ = ServiceLifecycle.reduce(&s, .removeRequested(deleteModel: true, managerAvailable: true))
        _ = ServiceLifecycle.reduce(&s, .removalResumed(deleteModel: true))
        #expect(ServiceLifecycle.reduce(&s, .shutdownServers) == [.detachTryIt, .stopServer(Self.current, wait: true)])
        #expect(ServiceLifecycle.reduce(&s, .removalFinished) == [.clearLog, .refresh, .familyChanged])
        _ = ServiceLifecycle.reduce(&s, .refreshed(.notInstalled, environment: .missing, requested: false, generation: s.generation))
        _ = ServiceLifecycle.reduce(&s, .exited(Self.current, status: 15, logTail: []))
        #expect(s.phase == .notInstalled)
    }

    // MARK: - The whole state space

    /// Every event, for every phase, with the server current or late (a stale
    /// id), and with or without a Stop/removal (`stopRequested`) and app quit
    /// (`shuttingDown`) under way. Each row names the property that must hold.
    @Test func everyEventAgainstEveryState() {
        let tail = ["last line"]
        for phase in Self.phases {
            for late in [false, true] {
                for stopping in [false, true] {
                    for quitting in [false, true] {
                        let id = late ? Self.stale : Self.current
                        let label = "\(phase) late=\(late) stop=\(stopping) quit=\(quitting)"
                        func run(_ event: LifecycleEvent, info: ServiceInfo = Service.tone.info, installing: Bool = false)
                            -> (before: LifecycleState, after: LifecycleState, effects: [LifecycleEffect])
                        {
                            let before = Self.state(
                                phase, info: info, stopRequested: stopping, shuttingDown: quitting, installing: installing)
                            var after = before
                            let effects = ServiceLifecycle.reduce(&after, event)
                            return (before, after, effects)
                        }

                        // A ready line flips only the current server's `.starting` to running.
                        let ready = run(.ready(id, port: 8665, pid: 9, rate: 24000, model: nil, at: Self.date))
                        if !late && phase == .starting {
                            #expect(ready.after.phase == .running && ready.effects == [.openTryIt(port: 8665)], "\(label)")
                        } else {
                            #expect(ready.after == ready.before && ready.effects.isEmpty, "ready \(label)")
                        }

                        // The ready timeout stops only the current server still starting.
                        let timeout = run(.readyTimedOut(id))
                        if !late && phase == .starting {
                            #expect(timeout.effects == [.stopServer(Self.current, wait: false)], "\(label)")
                        } else {
                            #expect(timeout.after == timeout.before && timeout.effects.isEmpty, "timeout \(label)")
                        }

                        // An exit: a late one changes nothing; a requested one only completes a Stop.
                        let exit = run(.exited(id, status: 1, logTail: tail))
                        if late {
                            #expect(exit.after == exit.before && exit.effects.isEmpty, "exit \(label)")
                        } else if stopping {
                            #expect(exit.after.phase == (phase == .stopping ? .installed : phase), "exit \(label)")
                        } else {
                            #expect(exit.after.phase == .failed(ServiceFailure(
                                kind: .crashed(status: 1), title: "Server stopped unexpectedly",
                                message: "The Tone server exited with status 1.", logTail: tail)), "exit \(label)")
                        }
                        if !late { #expect(exit.after.server == nil && exit.after.running == nil, "\(label)") }

                        // Start never runs during a removal or with a server up.
                        let start = run(.startRequested(appAvailable: true, managerAvailable: true, portFree: true))
                        #expect(start.after == start.before && start.effects.isEmpty, "start with a server \(label)")
                        var idle = Self.state(phase, server: nil)
                        let startIdle = ServiceLifecycle.reduce(
                            &idle, .startRequested(appAvailable: true, managerAvailable: true, portFree: true))
                        #expect(startIdle == (phase == .removing ? [] : [.launch(port: 8665)]), "start \(label)")

                        // Stop moves the current server to `.stopping` from any phase.
                        let stop = run(.stopRequested)
                        #expect(stop.after.phase == .stopping && stop.after.stopRequested, "stop \(label)")

                        // An install that ends while a removal owns the phase leaves it alone.
                        for outcome in [InstallOutcome.cancelled, .failed(stage: .model, message: "x", logTail: [])] {
                            let ended = run(.installEnded(outcome, then: .refresh), installing: true)
                            #expect(!ended.after.installing, "\(label)")
                            if phase == .removing {
                                #expect(ended.after.phase == .removing && ended.effects.isEmpty, "install \(label)")
                            }
                        }

                        // Quit cancelled the Apple Speech install: nothing to show.
                        for outcome in [InstallOutcome.cancelled, .failed(stage: .model, message: "x", logTail: [])] {
                            let ended = run(.appleSpeechInstallEnded(outcome), info: Service.appleSpeech.info, installing: true)
                            if quitting {
                                #expect(ended.after.phase == phase && ended.effects.isEmpty, "apple \(label)")
                            } else if outcome == .cancelled {
                                #expect(ended.after.phase == phase && ended.effects == [.refresh], "apple \(label)")
                            }
                        }

                        // A second Delete during a removal is a no-op; otherwise Delete takes the phase.
                        let remove = run(.removeRequested(deleteModel: true, managerAvailable: true))
                        if phase == .removing {
                            #expect(remove.after == remove.before && remove.effects.isEmpty, "remove \(label)")
                        } else {
                            #expect(remove.after.phase == .removing, "remove \(label)")
                            #expect(remove.effects.first == .cancelInstall, "remove \(label)")
                        }

                        // A sibling's change re-reads only settled phases.
                        let sibling = run(.environmentChanged(.missing))
                        let settled = [.checking, .notInstalled, .needsModel, .installed].contains(phase)
                        #expect(sibling.effects == (settled ? [.refresh] : []), "sibling \(label)")
                        #expect(sibling.after.environment == (settled ? sibling.before.environment : .missing), "\(label)")
                    }
                }
            }
        }
    }

    // MARK: - Async results against the state they land in

    /// Who owns the phase when a result lands.
    enum Owner: CaseIterable { case none, server, stoppedServer, native, install, probe }

    static func owned(_ phase: ServicePhase, by owner: Owner, info: ServiceInfo = Service.tone.info) -> LifecycleState {
        var s = state(
            phase, info: info, server: [.server, .stoppedServer].contains(owner) ? current : nil,
            stopRequested: owner == .stoppedServer, installing: owner == .install)
        if owner == .native { s.native = current }
        if owner == .probe { s.portProbe = 7 }
        return s
    }

    /// The contract, stated independently of the reducer: a refresh may move the
    /// phase only when nothing owns it.
    static func live(_ s: LifecycleState) -> Bool {
        if s.installing || (s.server != nil && !s.stopRequested) || s.native != nil || s.portProbe != nil { return true }
        return [.starting, .running, .stopping, .removing].contains(s.phase)
    }

    /// {phase} x {owner} x {refresh started under the current or an older
    /// generation} x {Python, Apple Speech}: a stale refresh changes nothing; a
    /// current one re-reads the environment but moves the phase only when no
    /// server, install, probe or removal owns it.
    @Test func everyRefreshAgainstEveryState() {
        for info in [Service.tone.info, Service.appleSpeech.info] {
            for phase in Self.phases {
                for owner in Owner.allCases where owner != .probe || phase == .starting {
                    for stale in [false, true] {
                        var s = Self.owned(phase, by: owner, info: info)
                        s.generation = 5
                        let before = s
                        let effects = ServiceLifecycle.reduce(
                            &s, .refreshed(.needsModel, environment: .missing, requested: false, generation: stale ? 4 : 5))
                        let label = "\(info.service) \(phase) owner=\(owner) stale=\(stale)"
                        #expect(effects.isEmpty, "\(label)")
                        #expect(s.generation == before.generation, "a refresh never bumps the generation: \(label)")
                        if stale {
                            #expect(s == before, "\(label)")
                        } else if Self.live(before) {
                            #expect(s.phase == before.phase && s.requested == before.requested, "\(label)")
                            #expect(s.environment == .missing, "\(label)")
                        } else {
                            #expect(s.phase == .needsModel && !s.requested && s.environment == .missing, "\(label)")
                        }
                    }
                }
            }
        }
    }

    /// Every non-refresh event that changes the phase or its owner bumps the
    /// generation (so a refresh in flight is dropped); one that changes neither
    /// leaves it, and a probe never outlives `.starting`.
    @Test func everyOwnershipChangeBumpsTheGeneration() {
        let events: [LifecycleEvent] = [
            .environmentChanged(.missing), .installRequested(managerAvailable: true), .cancelInstallRequested,
            .installStage(.packages), .installEnded(.finished, then: .refresh), .installEnded(.cancelled, then: .refresh),
            .installEnded(.failed(stage: .model, message: "x", logTail: []), then: .refresh),
            .appleSpeechInstallEnded(.finished), .startRequested(appAvailable: true, managerAvailable: true, portFree: false),
            .startRequested(appAvailable: true, managerAvailable: true, portFree: true),
            .portOwnerFound(PortProbe.Owner(), probe: 7), .launched(ServerID(3)), .launchFailed("x"), .nativeLaunched(ServerID(3)),
            .nativeBound(Self.current, port: 1, pid: 1, at: Self.date), .nativeFailed(Self.current, "x"),
            .ready(Self.current, port: 1, pid: 1, rate: 24000, model: nil, at: Self.date),
            .supervisedError(Self.current, SupervisedError(code: "x", message: nil)), .readyTimedOut(Self.current),
            .exited(Self.current, status: 1, logTail: []), .stopRequested, .shutdownRequested, .shutdownServers,
            .removeRequested(deleteModel: true, managerAvailable: true), .removalResumed(deleteModel: true),
            .removalFailed("x"), .removalFinished, .dismissFailure,
        ]
        for phase in Self.phases {
            for owner in Owner.allCases {
                for event in events {
                    var s = Self.owned(phase, by: owner)
                    if owner == .probe { s.phase = .starting }
                    let before = s
                    _ = ServiceLifecycle.reduce(&s, event)
                    let moved = s.phase != before.phase || s.server != before.server || s.native != before.native
                        || s.installing != before.installing
                    let label = "\(phase) owner=\(owner) \(event)"
                    #expect(s.generation == before.generation + (moved ? 1 : 0), "\(label)")
                    if s.phase != .starting { #expect(s.portProbe == nil, "\(label)") }
                }
            }
        }
    }

    /// {phase} x {probe token current, stale, none}: only the current probe of a
    /// start still `.starting` fails it as port-busy.
    @Test func everyPortOwnerAgainstEveryState() {
        for phase in Self.phases {
            for token in [Optional(7), 6, nil] {
                var s = Self.state(phase, server: nil)
                s.portProbe = phase == .starting ? token : nil
                let before = s
                _ = ServiceLifecycle.reduce(&s, .portOwnerFound(PortProbe.Owner(), probe: 7))
                if phase == .starting && token == 7 {
                    #expect(s.phase == .failed(.portBusy(port: 8665, owner: PortProbe.Owner())) && s.portProbe == nil, "\(phase)")
                } else {
                    #expect(s == before, "\(phase) token=\(String(describing: token))")
                }
            }
        }
    }

    /// An `error` line from a replaced server never reaches the current one.
    @Test func aStaleSupervisedErrorIsDropped() {
        for phase in Self.phases {
            var s = Self.state(phase)
            _ = ServiceLifecycle.reduce(&s, .supervisedError(Self.stale, SupervisedError(code: "EADDRINUSE", message: nil)))
            #expect(s.supervisedError == nil, "\(phase)")
            _ = ServiceLifecycle.reduce(&s, .exited(Self.current, status: 1, logTail: []))
            if case .failed(let f) = s.phase { #expect(f.kind == .crashed(status: 1), "\(phase)") }
        }
    }

    /// The reported sequences.
    @Test func startDuringARefreshKeepsTheStart() {
        var s = Self.state(.installed, server: nil)
        let g = s.generation  // a sibling's refresh begins
        _ = ServiceLifecycle.reduce(&s, .startRequested(appAvailable: true, managerAvailable: true, portFree: true))
        _ = ServiceLifecycle.reduce(&s, .launched(Self.current))
        _ = ServiceLifecycle.reduce(&s, .refreshed(.installed, generation: g))
        #expect(s.phase == .starting)
        _ = ServiceLifecycle.reduce(&s, .ready(Self.current, port: 8665, pid: 1, rate: 24000, model: nil, at: Self.date))
        #expect(s.phase == .running)
        _ = ServiceLifecycle.reduce(&s, .refreshed(.installed, generation: s.generation))  // Reset / Try Again
        #expect(s.phase == .running)
    }

    @Test func deleteDuringRefreshThenStartWins() {
        var s = Self.state(.installing(.packages), server: nil, installing: true)
        _ = ServiceLifecycle.reduce(&s, .installEnded(.finished, then: .refreshThenStart))
        let g = s.generation  // the refresh begins
        _ = ServiceLifecycle.reduce(&s, .removeRequested(deleteModel: false, managerAvailable: true))
        _ = ServiceLifecycle.reduce(&s, .refreshed(.installed, generation: g))
        #expect(s.phase == .removing)
        #expect(ServiceLifecycle.reduce(&s, .startRequested(appAvailable: true, managerAvailable: true, portFree: true)).isEmpty)
        _ = ServiceLifecycle.reduce(&s, .removalFinished)
        _ = ServiceLifecycle.reduce(&s, .refreshed(.installed, generation: g))  // the stale one, late
        #expect(s.phase == .checking)
        _ = ServiceLifecycle.reduce(&s, .refreshed(.notInstalled, generation: s.generation))
        #expect(s.phase == .notInstalled)
    }

    @Test func refreshKeepsARunningAppleSpeechServer() {
        var s = Self.state(.installed, info: Service.appleSpeech.info, server: nil)
        _ = ServiceLifecycle.reduce(&s, .nativeLaunched(Self.current))
        _ = ServiceLifecycle.reduce(&s, .nativeBound(Self.current, port: 9465, pid: 1, at: Self.date))
        #expect(s.phase == .running)
        _ = ServiceLifecycle.reduce(&s, .refreshed(.installed, generation: s.generation))
        #expect(s.phase == .running && s.native == Self.current)
    }

    @Test func cancelDuringThePortProbeDropsIt() {
        var s = Self.state(.installed, server: nil)
        let effects = ServiceLifecycle.reduce(&s, .startRequested(appAvailable: true, managerAvailable: true, portFree: false))
        #expect(effects.last == .probePortOwner(port: 8665, probe: 1) && s.portProbe == 1)
        #expect(ServiceLifecycle.reduce(&s, .stopRequested) == [.refresh])
        #expect(s.phase == .checking && s.portProbe == nil)
        _ = ServiceLifecycle.reduce(&s, .refreshed(.installed, generation: s.generation))
        _ = ServiceLifecycle.reduce(&s, .portOwnerFound(PortProbe.Owner(), probe: 1))
        #expect(s.phase == .installed)
        // Delete while probing: the late answer can't fail a removed service.
        _ = ServiceLifecycle.reduce(&s, .startRequested(appAvailable: true, managerAvailable: true, portFree: false))
        _ = ServiceLifecycle.reduce(&s, .removeRequested(deleteModel: true, managerAvailable: true))
        _ = ServiceLifecycle.reduce(&s, .portOwnerFound(PortProbe.Owner(), probe: 2))
        #expect(s.phase == .removing)
    }

    /// Apple Speech reported `.downloading` with no install of ours: attach to it.
    @Test func aSystemDownloadIsAttachedTo() {
        var s = Self.state(.checking, info: Service.appleSpeech.info, server: nil)
        #expect(ServiceLifecycle.reduce(&s, .refreshed(.installing(.model), generation: s.generation)) == [.runAppleSpeechInstall])
        #expect(s.installing && s.phase == .installing(.model))
        #expect(ServiceLifecycle.reduce(&s, .cancelInstallRequested) == [.cancelInstall])
        _ = ServiceLifecycle.reduce(&s, .appleSpeechInstallEnded(.cancelled))
        #expect(!s.installing)
    }

    /// The lifecycle's reading of itself (what the menu groups by).
    @Test func derivedFlagsForEveryPhase() {
        let runtime = ServiceFailure(kind: .crashed(status: 1), title: "", message: "")
        for phase in Self.phases + [.failed(runtime)] {
            for requested in [false, true] {
                var s = Self.state(phase, server: nil)
                s.requested = requested
                let active: Bool
                let installed: Bool
                let busy: Bool
                switch phase {
                case .starting, .running, .stopping: (active, installed) = (true, true)
                case .installed, .needsModel: (active, installed) = (false, true)
                case .installing, .removing: (active, installed) = (false, requested)
                case .failed(let f): (active, installed) = (f.isRuntime, f.isRuntime || requested)
                default: (active, installed) = (false, false)
                }
                switch phase {
                case .checking, .installing, .starting, .stopping, .removing: busy = true
                default: busy = false
                }
                #expect(s.isActive == active && s.isInstalled == installed && s.isBusy == busy, "\(phase) \(requested)")
                #expect(s.isInstalling == (phase == .installing(.model)), "\(phase)")
            }
        }
    }

    /// Quit cancels only an Apple Speech install, and waits for it boundedly.
    @Test func shutdownCancelsOnlyAnAppleSpeechInstall() {
        var apple = Self.state(.installing(.model), info: Service.appleSpeech.info, server: nil, installing: true)
        #expect(ServiceLifecycle.reduce(&apple, .shutdownRequested) == [.cancelInstall, .awaitInstall(bounded: true)])
        #expect(apple.shuttingDown)
        var python = Self.state(.installing(.model), server: nil, installing: true)
        #expect(ServiceLifecycle.reduce(&python, .shutdownRequested).isEmpty)
        #expect(!python.shuttingDown)
    }

    /// The native server: a bind that lands after Stop is discarded.
    @Test func aLateNativeBindIsDiscarded() {
        var s = Self.state(.installed, info: Service.appleSpeech.info, server: nil)
        _ = ServiceLifecycle.reduce(&s, .nativeLaunched(Self.current))
        #expect(s.phase == .starting)
        #expect(ServiceLifecycle.reduce(&s, .stopRequested) == [.stopNative, .tearDown(note: "stopped")])
        #expect(s.phase == .installed)
        #expect(ServiceLifecycle.reduce(&s, .nativeBound(Self.current, port: 9465, pid: 1, at: Self.date)) == [.discardNative(Self.current)])
        #expect(s.phase == .installed)
    }

    /// A supervised error decides how the exit reads.
    @Test func exitAfterSupervisedErrors() {
        var busy = Self.state(.starting)
        _ = ServiceLifecycle.reduce(&busy, .supervisedError(Self.current, SupervisedError(code: "EADDRINUSE", message: nil)))
        #expect(ServiceLifecycle.reduce(&busy, .exited(Self.current, status: 1, logTail: [])).last == .probePortOwner(port: 8665, probe: 1))
        var missing = Self.state(.starting)
        _ = ServiceLifecycle.reduce(&missing, .supervisedError(Self.current, SupervisedError(code: "ModuleNotFoundError", message: "m")))
        _ = ServiceLifecycle.reduce(&missing, .exited(Self.current, status: 1, logTail: []))
        #expect(missing.phase == .failed(ServiceFailure(kind: .startup(code: "ModuleNotFoundError"), title: "Environment incomplete", message: "m")))
    }

    /// The install's mode and first stage, and an outdated environment refreshed before start.
    @Test func installModesAndOutdatedStart() {
        #expect(ServiceLifecycle.installMode(environment: .missing, phase: .needsModel) == .full)
        #expect(ServiceLifecycle.installMode(environment: .ready(Self.marker), phase: .needsModel) == .model)
        let download = ServiceFailure(kind: .download, title: "", message: "")
        #expect(ServiceLifecycle.installMode(environment: .ready(Self.marker), phase: .failed(download)) == .model)
        #expect(ServiceLifecycle.firstStage(mode: .full, environment: .missing) == .python)
        var outdated = Self.state(.installed, server: nil)
        outdated.environment = .outdated(Self.marker)
        #expect(
            ServiceLifecycle.reduce(&outdated, .startRequested(appAvailable: true, managerAvailable: true, portFree: true))
                == [.runInstall(.packages, then: .refreshThenStart)])
        #expect(outdated.phase == .installing(.packages) && outdated.installing)
        #expect(ServiceLifecycle.reduce(&outdated, .installEnded(.finished, then: .refreshThenStart)) == [.refresh, .startIfNotCancelled, .familyChanged])
    }
}

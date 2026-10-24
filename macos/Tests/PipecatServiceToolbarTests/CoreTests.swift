import Foundation
import Testing

@testable import PipecatToolbarCore

@Suite struct LineSplitterTests {
    @Test func splitsOnNewlinesAndCarriageReturns() {
        var splitter = LineSplitter()
        #expect(splitter.feed(Data("a\nb\rc".utf8)) == ["a", "b"])
        #expect(splitter.feed(Data("d\r\n\ne".utf8)) == ["cd"])
        #expect(splitter.flush() == "e")
        #expect(splitter.flush() == nil)
    }

    @Test func handlesMultibyteSplitAcrossChunks() {
        var splitter = LineSplitter()
        let bytes = Array("héllo\n".utf8)
        #expect(splitter.feed(Data(bytes[0..<2])) == [])
        #expect(splitter.feed(Data(bytes[2...])) == ["héllo"])
    }
}

@Suite struct CatalogTests {
    /// TTS ports must match the tts repo's `justfile` `_resolve` table
    /// (`tests/test_macos_catalog_drift.py` checks the Swift source against it).
    @Test func canonicalPorts() {
        let expected: [Service: Int] = [
            .tone: 8665, .kokoro: 8765, .voxtralTTS: 8865, .pocketTTS: 8965,
            .dia: 9065, .qwen3TTS: 9165, .fishTTS: 9265, .phonon: 9365,
            .appleSpeech: 9465, .whisper: 9565, .kyutai: 9665,
            // Variants: the family's port + 1, 2, ...
            .phononQ8: 9366, .phononQ4: 9367,
            .whisperLargeV3: 9566, .whisperMedium: 9567, .whisperSmall: 9568, .whisperBase: 9569, .whisperTiny: 9570,
            .kyutai2B: 9666,
        ]
        for service in Service.allCases {
            #expect(service.info.port == expected[service], "\(service)")
        }
        #expect(Set(ServiceCatalog.all.map(\.port)).count == Service.allCases.count)
    }

    @Test func ttsFamilyNamesMatchServerBackendNames() {
        let families = ServiceCatalog.services(of: .tts).filter { $0.family == $0 }
        #expect(families.map(\.rawValue).sorted() == [
            "dia", "fish_tts", "kokoro", "phonon", "pocket_tts", "qwen3_tts", "tone", "voxtral_tts",
        ])
        #expect(ServiceCatalog.services(of: .stt).filter { $0.family == $0 } == [.appleSpeech, .whisper, .kyutai])
    }

    @Test func runtimes() {
        for service in ServiceCatalog.services(of: .tts) where service != .tone {
            #expect(service.info.runtime == .ttsServer(extra: service.family.rawValue))
            #expect(service.info.needsModelDownload)
        }
        #expect(Service.tone.info.runtime == .ttsServer(extra: nil))
        #expect(Service.appleSpeech.info.runtime == .appleSpeech)
        #expect(!Service.appleSpeech.info.runtime.usesPython)
        guard case .sttHelper(let engine, let reqs) = Service.kyutai.info.runtime else {
            Issue.record("kyutai runtime")
            return
        }
        #expect(engine == "kyutai" && reqs.contains("moshi_mlx==0.3.0"))
    }

    @Test func copyStrings() {
        #expect(Service.kokoro.info.endpoint == "ws://127.0.0.1:8765/")
        #expect(Service.kokoro.info.pipecatService() == #"LocalTTSService(host="127.0.0.1", port=8765)"#)
        #expect(Service.whisper.info.pipecatService() == #"LocalWebSocketSTTService(host="127.0.0.1", port=9565)"#)
        #expect(Service.whisper.info.environment() == "STT_WS_HOST=127.0.0.1 STT_WS_PORT=9565")
        #expect(Service.phonon.info.environment(port: 1234) == "TTS_WS_HOST=127.0.0.1 TTS_WS_PORT=1234")
    }

    @Test func restrictiveLicences() {
        #expect(Service.voxtralTTS.info.licence.isRestrictive)
        #expect(Service.fishTTS.info.licence.isRestrictive)
        #expect(!Service.kokoro.info.licence.isRestrictive)
        #expect(!Service.kyutai.info.licence.isRestrictive)
    }
}

@Suite struct VariantTests {
    @Test func families() {
        #expect(Service.phonon.variants == [.phonon, .phononQ8, .phononQ4])
        #expect(Service.whisperSmall.variants.first == .whisper)
        #expect(Service.kyutai2B.family == .kyutai)
        #expect(Service.kokoro.variants == [.kokoro])
        for service in Service.allCases {
            let info = service.info
            // Siblings share a runtime (one environment) and each has a label.
            #expect(info.runtime == service.family.info.runtime, "\(service)")
            #expect(info.kind == service.family.info.kind)
            #expect((info.variantLabel != nil) == (service.variants.count > 1), "\(service)")
        }
        #expect(Service.phononQ8.info.displayName == "Phonon 8-bit")
        #expect(Service.phonon.info.launchdLabel == "pipecat.tts-server.phonon")
        #expect(Service.phononQ8.info.launchdLabel == nil)
    }

    static func manager() throws -> (EnvironmentManager, AppPaths) {
        let root = FileManager.default.temporaryDirectory.appending(path: "toolbar-variants-\(UUID().uuidString.prefix(8))")
        // Model roots inside the temp dir too: a test never deletes from the real caches.
        let paths = AppPaths(
            supportDir: root, cacheDir: root.appending(path: "cache"), logsDir: root.appending(path: "logs"),
            hubDir: root.appending(path: "hub"), derivedModelsDir: root.appending(path: "derived"))
        try paths.prepare()
        let tools = Tools(
            uv: URL(filePath: "/usr/bin/true"), wheel: URL(filePath: "/tmp/pipecat_local_tts_server-0-py3-none-any.whl"),
            constraints: nil, helper: URL(filePath: "/tmp/helper.py"), source: .override)
        return (EnvironmentManager(paths: paths, tools: tools), paths)
    }

    @Test func commands() throws {
        let (manager, paths) = try Self.manager()
        let serve = try #require(manager.serveArguments(for: Service.phononQ8.info, port: 9366))
        #expect(Array(serve.prefix(7)) == ["-m", "tts_server", "serve", "--backend", "phonon", "--model", "phonon-q8"])
        #expect(manager.serveArguments(for: Service.phonon.info, port: 9365)?.contains("--model") == false)
        let prefetch = try #require(manager.prefetchArguments(for: Service.phononQ4.info, check: true))
        #expect(prefetch == ["-m", "tts_server", "prefetch", "--backend", "phonon", "--json", "--model", "phonon-q4", "--check"])

        let stt = try #require(manager.serveArguments(for: Service.whisperSmall.info, port: 9568))
        #expect(stt.suffix(2) == ["--model", "mlx-community/whisper-small-mlx"])
        let sttPrefetch = try #require(manager.prefetchArguments(for: Service.whisperSmall.info, check: false))
        #expect(sttPrefetch.contains("mlx-community/whisper-small-mlx"))
        #expect(sttPrefetch.contains(paths.envDir(.whisper).path))

        // One environment per family; logs per variant.
        #expect(paths.envDir(.whisperTiny) == paths.envDir(.whisper))
        #expect(paths.python(.phononQ8) == paths.python(.phonon))
        #expect(paths.appLog(.phononQ8) != paths.appLog(.phonon))
    }

    /// A built environment as `install` leaves it: an executable python and the marker.
    static func fakeEnvironment(_ service: Service, _ manager: EnvironmentManager, _ paths: AppPaths) async throws {
        try FileManager.default.createDirectory(
            at: paths.python(service).deletingLastPathComponent(), withIntermediateDirectories: true)
        FileManager.default.createFile(
            atPath: paths.python(service).path, contents: Data("#!/bin/sh\n".utf8), attributes: [.posixPermissions: 0o755])
        try EnvironmentMarker(
            service: service.rawValue, fingerprint: await manager.tools.fingerprint(for: service.info), python: "3.12.0",
            createdAt: Date()
        ).write(paths.marker(service))
    }

    @Test func siblingsShareAnEnvironment() async throws {
        let (manager, paths) = try Self.manager()
        try await Self.fakeEnvironment(.phonon, manager, paths)
        // Built before variants existed: the marker names the one service.
        #expect(manager.isRequested(.phonon))
        #expect(!manager.isRequested(.phononQ8))
        guard case .ready = manager.environmentStatus(.phononQ8) else {
            Issue.record("a sibling sees the family environment")
            return
        }

        try await manager.recordVariant(.phononQ8, installed: true)
        #expect(manager.isRequested(.phonon) && manager.isRequested(.phononQ8))
        #expect(!manager.isRequested(.phononQ4))

        // Removing one variant keeps the environment for the other...
        try await manager.remove(.phonon, deleteModel: false)
        #expect(!manager.isRequested(.phonon) && manager.isRequested(.phononQ8))
        #expect(manager.environmentStatus(.phononQ8) != .missing)
        // ...and the last one takes it with it.
        try await manager.remove(.phononQ8, deleteModel: false)
        #expect(manager.environmentStatus(.phonon) == .missing)
        #expect(!FileManager.default.fileExists(atPath: paths.envDir(.phonon).path))
    }

    @Test func deletesOnlyUnsharedModelFiles() {
        let hub = URL(filePath: "/Users/x/.cache/huggingface/hub")
        let repo = hub.appending(path: "models--gradium--phonon-31m-en")
        let voices = hub.appending(path: "models--gradium--phonon-voices")
        let q8 = URL(filePath: "/Users/x/.cache/pipecat-tts-server/phonon/52433c46975b-q8-g32-mlxaudio0.6.1")
        // Phonon full precision is still installed: it keeps the repo and voices.
        #expect(
            EnvironmentManager.unshared(
                [repo, voices, q8], siblingRepos: ["gradium/phonon-31m-en"], siblingFolders: [voices]) == [q8])
        #expect(EnvironmentManager.unshared([repo], siblingRepos: [], siblingFolders: []) == [repo])

        let derived = URL(filePath: "/Users/x/.cache/pipecat-tts-server")
        #expect(EnvironmentManager.isDeletableModelPath(repo, hub: hub, derived: derived))
        #expect(EnvironmentManager.isDeletableModelPath(q8, hub: hub, derived: derived))
        #expect(!EnvironmentManager.isDeletableModelPath(URL(filePath: "/Users/x/Documents"), hub: hub, derived: derived))
        #expect(!EnvironmentManager.isDeletableModelPath(derived, hub: hub, derived: derived))
    }

    /// The delete guard's whole contract: a candidate is deletable iff it is a
    /// `models--*` folder directly in the hub, or `<derived>/<backend>/<key>`.
    @Test func deletionStaysInsideTheModelRoots() {
        let hub = URL(filePath: "/Users/x/.cache/huggingface/hub")
        let derived = URL(filePath: "/Users/x/.cache/pipecat-tts-server")
        let cases: [(String, Bool)] = [
            ("/Users/x/.cache/huggingface/hub/models--org--name", true),
            ("/Users/x/.cache/huggingface/hub/models--org--name/", true),
            ("/Users/x/.cache/pipecat-tts-server/phonon/abc-q8", true),
            // Look-alike names outside the roots.
            ("/Users/x/Documents/models--archive", false),
            ("/Users/x/models--x", false),
            ("/Users/x/.cache/huggingface/hub/sub/models--org--name", false),
            ("/Users/x/Documents/pipecat-tts-server/phonon/abc", false),
            ("/tmp/pipecat-tts-server/a/b", false),
            // Not a repo folder, or the roots themselves.
            ("/Users/x/.cache/huggingface/hub", false),
            ("/Users/x/.cache/huggingface/hub/models--", false),
            ("/Users/x/.cache/huggingface/hub/datasets--org--name", false),
            ("/Users/x/.cache/huggingface/hub/.locks", false),
            ("/Users/x/.cache/pipecat-tts-server", false),
            ("/Users/x/.cache/pipecat-tts-server/phonon", false),
            ("/Users/x/.cache/pipecat-tts-server/phonon/abc/deeper", false),
            ("/Users/x/.cache/pipecat-tts-server/.hidden/abc", false),
            // `..` that resolves out of (or back into) a root.
            ("/Users/x/.cache/huggingface/hub/../../../Documents/models--x", false),
            ("/Users/x/.cache/pipecat-tts-server/phonon/../../../../Documents", false),
            ("/Users/x/Documents/../.cache/huggingface/hub/models--org--name", true),
        ]
        for (path, deletable) in cases {
            #expect(
                EnvironmentManager.isDeletableModelPath(URL(filePath: path), hub: hub, derived: derived) == deletable,
                "\(path)")
        }
    }

    @Test func deletionRootsAreComparedWithSymlinksResolved() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "toolbar-links-\(UUID().uuidString.prefix(8))")
        let real = root.appending(path: "real-hub")
        try FileManager.default.createDirectory(at: real.appending(path: "models--org--name"), withIntermediateDirectories: true)
        let link = root.appending(path: "hub-link")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: real)
        let elsewhere = root.appending(path: "elsewhere")
        #expect(EnvironmentManager.isDeletableModelPath(link.appending(path: "models--org--name"), hub: real, derived: elsewhere))
        #expect(EnvironmentManager.isDeletableModelPath(real.appending(path: "models--org--name"), hub: link, derived: elsewhere))
        #expect(!EnvironmentManager.isDeletableModelPath(root.appending(path: "models--org--name"), hub: link, derived: elsewhere))
    }

    @Test func quantizedVariantNeedsItsConvertedWeights() throws {
        let check = #"{"event":"check","repo":"gradium/phonon-31m-en","cached":true,"path":"/h/hub/models--gradium--phonon-31m-en/snapshots/abc","size_bytes":10}"#
        let path = "/c/pipecat-tts-server/phonon/abc-q8"
        let notYet = try #require(
            EnvironmentManager.cacheStatus(checkOutput: [
                check,
                #"{"event":"summary","cached":true,"size_bytes":10,"quantized":false,"quantized_path":"\#(path)"}"#,
            ]))
        #expect(!notYet.cached && notYet.derivedFolders.isEmpty)
        #expect(notYet.repoFolders.map(\.lastPathComponent) == ["models--gradium--phonon-31m-en"])
        let converted = try #require(
            EnvironmentManager.cacheStatus(checkOutput: [
                check, #"{"event":"summary","cached":true,"size_bytes":10,"quantized":true,"quantized_path":"\#(path)"}"#,
            ]))
        #expect(converted.cached && converted.derivedFolders == [URL(filePath: path)])
        // Unquantized backends report no `quantized` field.
        #expect(
            EnvironmentManager.cacheStatus(checkOutput: [check, #"{"event":"summary","cached":true,"size_bytes":10}"#])?
                .cached == true)
    }
}

@Suite struct EventDecodingTests {
    @Test func prefetchProgress() throws {
        let line = #"{"event":"progress","repo":"r","downloaded_bytes":10,"total_bytes":40,"files_done":1,"files_total":5}"#
        let event = try #require(ServerCLIEvent.parse(line))
        #expect(event.event == "progress")
        #expect(event.downloadedBytes == 10 && event.totalBytes == 40)
        #expect(event.filesDone == 1 && event.filesTotal == 5)
    }

    @Test func readyLine() throws {
        let line = #"{"event":"ready","backend":"tone","model":null,"host":"127.0.0.1","port":8665,"socket_path":null,"rate":24000,"pid":42}"#
        let event = try #require(ServerCLIEvent.parse(line))
        #expect(event.port == 8665 && event.rate == 24000 && event.pid == 42)
    }

    @Test func nonJSONIsIgnored() {
        #expect(ServerCLIEvent.parse("Fetching 5 files: 100%") == nil)
    }

    @Test func downloadFraction() {
        #expect(DownloadProgress(repo: "r").fraction == nil)
        #expect(DownloadProgress(repo: "r", filesDone: 1, filesTotal: 4).fraction == 0.25)
        #expect(DownloadProgress(repo: "r", downloadedBytes: 50, totalBytes: 100, filesDone: 4, filesTotal: 4).fraction == 0.5)
    }

    @Test func helloAndAudio() throws {
        let hello = #"{"type":"server.hello","backend":{"name":"tone","model":null},"audio":{"format":"pcm16","rate":24000,"channels":1},"capabilities":{"streaming":false,"voice_count":0,"max_text_chars":4096}}"#
        guard case .hello(let h)? = TTSEvent.decode(hello) else { Issue.record("not hello"); return }
        #expect(h.rate == 24000 && h.backend == "tone" && h.maxTextChars == 4096)

        let pcm = Data([0x00, 0x80, 0xFF, 0x7F])  // -32768, 32767
        let delta = #"{"type":"response.audio.delta","response_id":"r1","seq":3,"audio":"\#(pcm.base64EncodedString())"}"#
        guard case .audio(let rid, let seq, let data)? = TTSEvent.decode(delta) else {
            Issue.record("not audio")
            return
        }
        #expect(rid == "r1" && seq == 3 && data == pcm)
    }

    @Test func pcmConversionExtremes() {
        let floats = PCM16.floats(from: Data([0x00, 0x80, 0xFF, 0x7F, 0x00, 0x00]))
        #expect(floats[0] == -1.0)
        #expect(abs(floats[1] - 0.99997) < 0.0001)
        #expect(floats[2] == 0)
    }

    @Test func snapshotToRepoFolder() {
        let folder = EnvironmentManager.repoFolder(
            fromSnapshot: "/u/.cache/huggingface/hub/models--gradium--phonon-31m-en/snapshots/abc")
        #expect(folder?.lastPathComponent == "models--gradium--phonon-31m-en")
        #expect(EnvironmentManager.repoFolder(fromSnapshot: "/tmp/some/dir") == nil)
    }
}

@Suite struct SubprocessTests {
    @Test func collectsLinesAndStatus() async throws {
        let sub = Subprocess(
            .init(
                executable: URL(filePath: "/bin/sh"),
                arguments: ["-c", "printf 'one\\ntwo'; echo err >&2; exit 3"],
                environment: ProcessEnvironment.base()))
        let result = try await sub.collect()
        #expect(result.status == 3)
        #expect(result.stdout == ["one", "two"])
        #expect(result.stderr == ["err"])
    }

    @Test func heldStdinEOFEndsChild() async throws {
        // `cat` exits when its stdin closes: the same contract as `serve --supervised`.
        let sub = Subprocess(
            .init(
                executable: URL(filePath: "/bin/cat"), arguments: [],
                environment: ProcessEnvironment.base(), holdStdin: true))
        try sub.run()
        try await Task.sleep(for: .milliseconds(200))
        #expect(sub.isRunning)
        sub.closeStdin()
        #expect(await sub.waitForExit() == 0)
    }

    @Test func stopEscalatesToKill() async throws {
        let sub = Subprocess(
            .init(
                executable: URL(filePath: "/bin/sh"), arguments: ["-c", "trap '' TERM; sleep 30"],
                environment: ProcessEnvironment.base()))
        try sub.run()
        try await Task.sleep(for: .milliseconds(200))
        let clock = ContinuousClock()
        let start = clock.now
        _ = await sub.stop(grace: .milliseconds(500))
        #expect(!sub.isRunning)
        #expect(clock.now - start < .seconds(5))
    }
}

/// End to end against a real `serve --supervised` tone server, when the repo's
/// dev environment exists (`uv sync --extra client` at the repo root).
@Suite struct ToneServerIntegrationTests {
    static let repo = URL(filePath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent()
    static let python = repo.appending(path: ".venv/bin/python")

    @Test(.enabled(if: FileManager.default.isExecutableFile(atPath: python.path)))
    func supervisedToneServerSpeaks() async throws {
        let server = Subprocess(
            .init(
                executable: Self.python,
                arguments: ["-m", "tts_server", "serve", "--backend", "tone", "--host", "127.0.0.1", "--port", "0", "--supervised"],
                environment: ProcessEnvironment.base(), currentDirectory: Self.repo, holdStdin: true))
        try server.run()
        defer { server.kill() }

        var port: Int?
        for await line in server.stdout {
            if let event = ServerCLIEvent.parse(line), event.event == "ready" {
                port = event.port
                break
            }
        }
        let readyPort = try #require(port)

        // No 403: URLSession must not send an Origin header.
        let client = TTSClient(port: readyPort)
        let hello = try await client.connect()
        #expect(hello.backend == "tone" && hello.rate == 24000)

        try await client.speak("Hello there.", voice: nil, eventID: "tryit-1")
        var frames = 0
        var finished = false
        while let event = try await client.next() {
            if case .audio = event { frames += 1 }
            if case .done = event { finished = true; break }
            if case .failed(_, let message) = event { Issue.record("failed: \(message)"); break }
        }
        #expect(finished && frames > 0)
        await client.close()

        server.closeStdin()
        #expect(await server.waitForExit() == 0)
    }
}

@Suite struct LogoTests {
    @Test func tokenizesImplicitSeparators() {
        #expect(SVGPath.tokenize("M45.7 0.76C50.4-0.99") == [
            .command("M"), .number(45.7), .number(0.76), .command("C"), .number(50.4), .number(-0.99),
        ])
    }

    @Test func markFillsItsViewBox() {
        let box = PipecatLogo.path.boundingBoxOfPath
        #expect(abs(box.minX) < 0.5 && abs(box.minY) < 0.5)
        #expect(abs(box.maxX - 332) < 0.5 && abs(box.maxY - 192) < 0.5)
    }

    @Test func fittingKeepsAspectAndCentres() {
        let fitted = PipecatLogo.path(fitting: CGRect(x: 0, y: 0, width: 100, height: 100)).boundingBoxOfPath
        #expect(abs(fitted.width - 100) < 0.5)
        #expect(abs(fitted.height - 100 * 192 / 332) < 0.5)
        #expect(abs(fitted.midY - 50) < 0.5)
    }
}


@Suite struct STTServerTests {
    static func fakeServer() -> STTWebSocketServer {
        STTWebSocketServer(identity: .init(name: "fake", model: nil)) { pcm, language in
            "heard \(pcm.count) bytes" + (language.map { " in \($0)" } ?? "")
        }
    }

    @Test func clientRoundTrip() async throws {
        let server = Self.fakeServer()
        let port = try await server.start(port: 0)
        defer { server.stop() }
        let client = STTClient(port: port)
        let hello = try await client.connect()
        #expect(hello.backend == "fake" && hello.rate == 16000)
        #expect(try await client.transcribe(Data(count: 32000)) == "heard 32000 bytes")
        // A second utterance on the same connection.
        #expect(try await client.transcribe(Data(count: 1000), chunkBytes: 300) == "heard 1000 bytes")
        await client.close()
    }

    @Test func busyPortFails() async throws {
        let first = Self.fakeServer()
        let port = try await first.start(port: 0)
        defer { first.stop() }
        await #expect(throws: STTWebSocketServer.StartError.self) {
            _ = try await Self.fakeServer().start(port: port)
        }
    }

    @Test func browserOriginRejected() async throws {
        let server = Self.fakeServer()
        let port = try await server.start(port: 0)
        defer { server.stop() }
        var request = URLRequest(url: URL(string: "ws://127.0.0.1:\(port)/")!)
        request.setValue("https://evil.example", forHTTPHeaderField: "Origin")
        let task = URLSession(configuration: .ephemeral).webSocketTask(with: request)
        task.resume()
        await #expect(throws: (any Error).self) { _ = try await task.receive() }
    }
}

@Suite struct LifecycleTests {
    /// A `uv` stand-in that records its pid and then blocks, like a long download.
    static func slowUV(in dir: URL) throws -> (uv: URL, pidFile: URL) {
        let pidFile = dir.appending(path: "uv.pid")
        let uv = dir.appending(path: "uv")
        try "#!/bin/sh\necho $$ > '\(pidFile.path)'\nexec sleep 60\n".write(to: uv, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: uv.path)
        return (uv, pidFile)
    }

    static func alive(_ pid: Int32) -> Bool { kill(pid, 0) == 0 }

    @Test func shutdownStopsAnInFlightInstall() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "toolbar-quit-\(UUID().uuidString.prefix(8))")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let paths = AppPaths(supportDir: root, cacheDir: root.appending(path: "cache"), logsDir: root.appending(path: "logs"))
        try paths.prepare()
        let (uv, pidFile) = try Self.slowUV(in: root)
        let tools = Tools(
            uv: uv, wheel: URL(filePath: "/tmp/pipecat_local_tts_server-0-py3-none-any.whl"), constraints: nil, helper: nil,
            source: .override)
        let manager = EnvironmentManager(paths: paths, tools: tools)

        let install = Task { () -> Error? in
            do {
                for try await _ in manager.install(.tone) {}
                return nil
            } catch { return error }
        }
        var pid: Int32?
        for _ in 0..<200 where pid == nil {  // up to ~10 s for the stand-in to start
            pid = (try? String(contentsOf: pidFile, encoding: .utf8)).flatMap { Int32($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
            if pid == nil { try await Task.sleep(for: .milliseconds(50)) }
        }
        let installer = try #require(pid, "the installer subprocess never started")
        #expect(Self.alive(installer))

        await manager.shutdown(grace: .seconds(2))

        // Gone before shutdown returned, and the install unwound as a cancellation, not a hang.
        #expect(!Self.alive(installer))
        #expect(await install.value is CancellationError)
        // Nothing can start after shutdown.
        var late: Error?
        do { for try await _ in manager.install(.tone) {} } catch { late = error }
        #expect(late is CancellationError)
    }

    @Test func removingAnOutdatedEnvironmentDeletesItsModel() async throws {
        let (manager, paths) = try VariantTests.manager()
        let hub = paths.supportDir.appending(path: "hub")
        let repo = hub.appending(path: "models--mlx-community--pocket-tts")
        let snapshot = repo.appending(path: "snapshots/abc")
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: snapshot.appending(path: "w.safetensors").path, contents: Data(count: 8))

        // The environment's python answers `prefetch --check --json` like the real one.
        let python = paths.python(.pocketTTS)
        try FileManager.default.createDirectory(at: python.deletingLastPathComponent(), withIntermediateDirectories: true)
        let json = """
            {"event":"check","path":"\(snapshot.path)","cached":true}
            {"event":"summary","cached":true,"size_bytes":8,"path":"\(snapshot.path)"}
            """
        try "#!/bin/sh\ncat <<'EOF'\n\(json)\nEOF\n".write(to: python, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: python.path)
        // Installed by an older app build: the fingerprint no longer matches.
        try EnvironmentMarker(service: Service.pocketTTS.rawValue, fingerprint: "old.whl", python: "3.12.0", createdAt: Date())
            .write(paths.marker(.pocketTTS))
        guard case .outdated = manager.environmentStatus(.pocketTTS) else {
            Issue.record("expected an outdated environment")
            return
        }

        try await manager.remove(.pocketTTS, deleteModel: true)

        #expect(manager.environmentStatus(.pocketTTS) == .missing)
        #expect(!FileManager.default.fileExists(atPath: repo.path), "model weights survived removal")
    }

    @Test func modelFilesThatCannotBeDeletedAreReported() async throws {
        let (manager, paths) = try VariantTests.manager()
        let repo = paths.supportDir.appending(path: "hub/models--org--stuck")
        try FileManager.default.createDirectory(at: repo.appending(path: "snapshots/abc"), withIntermediateDirectories: true)
        // A read-only parent makes the delete fail.
        try FileManager.default.setAttributes([.posixPermissions: 0o555], ofItemAtPath: repo.deletingLastPathComponent().path)
        defer { try? FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: repo.deletingLastPathComponent().path) }
        await #expect(throws: RemovalError.self) {
            try await manager.remove(.pocketTTS, deleteModel: true, knownFolders: [repo])
        }
    }

    @Test func hubCacheFollowsHuggingFaceVariables() {
        let home = URL(filePath: "/Users/x")
        #expect(AppPaths.huggingFaceHub(environment: [:], home: home).path == "/Users/x/.cache/huggingface/hub")
        #expect(AppPaths.huggingFaceHub(environment: ["XDG_CACHE_HOME": "/c"], home: home).path == "/c/huggingface/hub")
        #expect(AppPaths.huggingFaceHub(environment: ["HF_HOME": "/hf", "XDG_CACHE_HOME": "/c"], home: home).path == "/hf/hub")
        #expect(AppPaths.huggingFaceHub(environment: ["HF_HUB_CACHE": "/hub", "HF_HOME": "/hf"], home: home).path == "/hub")
    }
}

/// A flag any task can set and poll (for "did it finish in time" checks that
/// must not hang the test when the answer is no).
final class Flag: @unchecked Sendable {
    private let lock = NSLock()
    private var set = false
    var value: Bool { lock.withLock { set } }
    func raise() { lock.withLock { set = true } }

    /// Polls for up to `limit`.
    func wait(_ limit: Duration) async -> Bool {
        let deadline = ContinuousClock.now + limit
        while !value, ContinuousClock.now < deadline { try? await Task.sleep(for: .milliseconds(20)) }
        return value
    }
}

final class Lines: @unchecked Sendable {
    private let lock = NSLock()
    private var lines: [String] = []
    func append(_ line: String) { lock.withLock { lines.append(line) } }
    var all: [String] { lock.withLock { lines } }
}

@Suite struct SupervisedExitOrderingTests {
    /// The exit lands first and the child's last line after it (the order the
    /// two pipes' queues can deliver them in): the line is still read before
    /// the exit is reported.
    @Test func exitIsReportedAfterTheFinalLine() async throws {
        let sub = Subprocess(
            .init(
                executable: URL(filePath: "/bin/sh"),
                arguments: ["-c", #"(sleep 0.3; printf '{"event":"error","code":"EADDRINUSE"}\n') & exit 1"#],
                environment: ProcessEnvironment.base()))
        try sub.run()
        let lines = Lines()
        let reader = Task { for await line in sub.stdout { lines.append(line) } }
        let status = await sub.waitForExit(after: reader)
        #expect(status == 1)
        #expect(lines.all.compactMap(ServerCLIEvent.parse).last?.errorCode == .addressInUse)
    }

    @Test func drainIsBoundedWhenThePipeStaysOpen() async throws {
        let sub = Subprocess(
            .init(executable: URL(filePath: "/bin/sh"), arguments: ["-c", "(sleep 30) & exit 0"], environment: ProcessEnvironment.base()))
        try sub.run()
        let reader = Task { for await _ in sub.stdout {} }
        let clock = ContinuousClock()
        let start = clock.now
        #expect(await sub.waitForExit(after: reader, drainLimit: .milliseconds(300)) == 0)
        #expect(clock.now - start < .seconds(5))
    }
}

@Suite struct StreamConsumeTests {
    enum Boom: Error { case boom }

    /// How a consumer ends, for every way the stream or the consumer can end.
    @Test func everyEndingIsReportedAsItHappened() async throws {
        enum Ending { case finished, threw, cancelled }
        for ending in [Ending.finished, .threw, .cancelled] {
            let (stream, continuation) = AsyncThrowingStream.makeStream(of: Int.self, throwing: (any Error).self)
            continuation.yield(1)
            let seen = Lines()
            let consumer = Task { () -> (any Error)? in
                do {
                    try await stream.consume { seen.append("\($0)") }
                    return nil
                } catch { return error }
            }
            while seen.all.isEmpty { try await Task.sleep(for: .milliseconds(10)) }
            switch ending {
            case .finished: continuation.finish()
            case .threw: continuation.finish(throwing: Boom.boom)
            case .cancelled: consumer.cancel()
            }
            let error = await consumer.value
            switch ending {
            case .finished: #expect(error == nil)
            case .threw: #expect(error is Boom)
            // The bug: a cancelled `for try await` returns normally, like `.finished`.
            case .cancelled: #expect(error is CancellationError)
            }
            // Nothing is delivered after a cancel.
            continuation.yield(2)
            #expect(seen.all == ["1"], "\(ending)")
        }
    }
}

@Suite struct ResponseTrackerTests {
    static let busy = "The server is busy. Try again in a moment."

    /// Request 1 played as `r1` and was stopped; request 2 is current as `r2`.
    static func twoRequests() -> (ResponseTracker, old: String, new: String) {
        var t = ResponseTracker()
        let old = t.committed()
        _ = t.route(.committed(responseID: "r1", previousEventID: old), speaking: true)
        t.abandon()
        let new = t.committed()
        _ = t.route(.committed(responseID: "r2", previousEventID: new), speaking: true)
        return (t, old, new)
    }

    @Test func staleCancelAfterStopDoesNotStopTheNextUtterance() {
        var t = ResponseTracker()
        let first = t.committed()
        _ = t.route(.committed(responseID: "r1", previousEventID: first), speaking: true)
        #expect(t.isCurrent("r1"))
        t.abandon()  // Stop
        let second = t.committed()  // Speak again before r1's cancel acknowledgement
        #expect(t.route(.cancelled(responseID: "r1"), speaking: true) == .ignore)
        _ = t.route(.committed(responseID: "r2", previousEventID: second), speaking: true)
        #expect(t.isCurrent("r2") && !t.isCurrent("r1"))
    }

    /// The reported bug: Stop abandons a commit before its refusal arrives; the
    /// late refusal must not fail the next request.
    @Test func lateRefusalOfAStoppedRequestIsIgnored() {
        var t = ResponseTracker()
        let first = t.committed()
        t.abandon()
        let second = t.committed()
        #expect(t.route(.error(code: "busy", message: "", previousEventID: first), speaking: true) == .ignore)
        _ = t.route(.committed(responseID: "r2", previousEventID: second), speaking: true)
        #expect(t.route(.audio(responseID: "r2", seq: 0, pcm: Data([1, 2])), speaking: true) == .play(Data([1, 2])))
    }

    /// Every event type x request ownership x speaking, against the table.
    @Test func everyEventIsRoutedByItsOwner() {
        enum Own: CaseIterable { case current, stale, none }
        enum Kind: CaseIterable { case audio, done, cancelled, failed, busy, refused, appendRefused }
        let pcm = Data([0, 1])
        for own in Own.allCases {
            for kind in Kind.allCases {
                for speaking in [true, false] {
                    var (t, old, new) = Self.twoRequests()
                    let rid: String? = own == .current ? "r2" : own == .stale ? "r1" : nil
                    let request: String? = own == .current ? new : own == .stale ? old : nil
                    let event: TTSEvent
                    switch kind {
                    case .audio: event = .audio(responseID: rid ?? "", seq: 0, pcm: pcm)
                    case .done: event = .done(responseID: rid ?? "", durationMs: nil)
                    case .cancelled: event = .cancelled(responseID: rid ?? "")
                    case .failed: event = .failed(responseID: rid, message: "boom")
                    case .busy: event = .error(code: "busy", message: "backlog", previousEventID: request)
                    case .refused: event = .error(code: "invalid_config", message: "bad", previousEventID: request)
                    case .appendRefused:
                        event = .error(
                            code: "payload_too_large", message: "long", previousEventID: request.map(ResponseTracker.appendID))
                    }
                    let expected: TryItAction
                    switch (kind, own, speaking) {
                    case (.audio, .current, true): expected = .play(pcm)
                    case (.done, .current, true): expected = .finish
                    case (.cancelled, .current, _): expected = .stopped
                    case (.failed, .current, _), (.failed, .none, true): expected = .fail("boom")
                    case (.busy, .current, _), (.busy, .none, true): expected = .fail(Self.busy)
                    case (.refused, .current, _), (.refused, .none, true): expected = .fail("bad")
                    case (.appendRefused, .current, _), (.appendRefused, .none, true): expected = .fail("long")
                    default: expected = .ignore
                    }
                    #expect(t.route(event, speaking: speaking) == expected, "\(kind) \(own) speaking=\(speaking)")
                }
            }
        }
    }

    /// Every sequence of up to 6 operations, against a direct model: the server
    /// answers commits in order (accepting or refusing each), echoing its id, and
    /// only the newest, un-stopped commit counts.
    @Test func matchesTheCommitModelForEverySequence() {
        enum Op: CaseIterable { case commit, accept, refuse, abandon }
        func check(_ ops: [Op]) {
            var tracker = ResponseTracker()
            var pending: [(index: Int, id: String)] = []
            var wanted: Int?
            var currentCommit: Int?
            var commits = 0
            for op in ops {
                switch op {
                case .commit:
                    let id = tracker.committed()
                    pending.append((commits, id))
                    wanted = commits
                    currentCommit = nil
                    commits += 1
                case .accept:
                    guard !pending.isEmpty else { return }  // the server never does this
                    let (index, id) = pending.removeFirst()
                    if index == wanted { currentCommit = index }
                    #expect(tracker.route(.committed(responseID: "r\(index)", previousEventID: id), speaking: true) == .ignore)
                case .refuse:
                    guard !pending.isEmpty else { return }
                    let (index, id) = pending.removeFirst()
                    let action = tracker.route(.error(code: "invalid_config", message: "no", previousEventID: id), speaking: true)
                    #expect(action == (index == wanted ? .fail("no") : .ignore), "\(ops)")
                case .abandon:
                    wanted = nil
                    currentCommit = nil
                    tracker.abandon()
                }
                #expect(tracker.current == currentCommit.map { "r\($0)" }, "\(ops)")
                for index in 0..<commits where index != currentCommit {
                    #expect(!tracker.isCurrent("r\(index)"), "\(ops)")
                }
            }
        }
        var sequences: [[Op]] = [[]]
        for _ in 0..<6 {
            sequences = sequences.flatMap { seq in Op.allCases.map { seq + [$0] } }
            sequences.forEach(check)
        }
    }
}

@Suite struct STTServerStartupTests {
    /// Stop while the listener is still coming up: start() must end (with an
    /// error, or the port if it bound first), never hang.
    @MainActor @Test func stopDuringStartupEndsTheStart() async throws {
        for _ in 0..<20 {
            let server = STTServerTests.fakeServer()
            let started = Flag()
            let start = Task { @MainActor in
                defer { started.raise() }
                _ = try await server.start(port: 0)
            }
            await Task.yield()
            server.stop()
            let ended = await started.wait(.seconds(3))
            #expect(ended, "start() never returned after stop()")
            guard ended else { return }
            _ = await start.result
        }
    }
}

@Suite struct InstallCancellationTests {
    static func paths() throws -> AppPaths {
        let root = FileManager.default.temporaryDirectory.appending(path: "toolbar-cancel-\(UUID().uuidString.prefix(8))")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let paths = AppPaths(
            supportDir: root, cacheDir: root.appending(path: "cache"), logsDir: root.appending(path: "logs"),
            hubDir: root.appending(path: "hub"), derivedModelsDir: root.appending(path: "derived"))
        try paths.prepare()
        return paths
    }

    /// An installer that ignores SIGTERM is killed after the grace period, so
    /// Cancel always ends the install (and frees the family lock).
    @Test func cancelKillsAnInstallerThatIgnoresSigterm() async throws {
        let paths = try Self.paths()
        let pidFile = paths.supportDir.appending(path: "uv.pid")
        let uv = paths.supportDir.appending(path: "uv")
        try "#!/bin/sh\ntrap '' TERM\necho $$ > '\(pidFile.path)'\nexec sleep 30\n".write(to: uv, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: uv.path)
        let manager = EnvironmentManager(
            paths: paths,
            tools: Tools(uv: uv, wheel: URL(filePath: "/tmp/x.whl"), constraints: nil, helper: nil, source: .override),
            cancelGrace: .milliseconds(500))

        let ended = Flag()
        let install = Task {
            defer { ended.raise() }
            try await manager.install(.tone).consume { _ in }
        }
        var pid: Int32?
        for _ in 0..<200 where pid == nil {
            pid = (try? String(contentsOf: pidFile, encoding: .utf8)).flatMap { Int32($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
            if pid == nil { try await Task.sleep(for: .milliseconds(50)) }
        }
        let installer = try #require(pid, "the installer subprocess never started")
        install.cancel()
        #expect(await ended.wait(.seconds(5)), "the cancelled install never ended")
        let dead = Flag()
        Task {
            while LifecycleTests.alive(installer) { try? await Task.sleep(for: .milliseconds(20)) }
            dead.raise()
        }
        #expect(await dead.wait(.seconds(5)), "the installer survived Cancel")
        // The family lock is free again: the next install gets past `.queued`.
        let reached = Flag()
        let again = Task {
            try? await manager.install(.tone).consume { if $0 == .stage(.python) { reached.raise() } }
        }
        #expect(await reached.wait(.seconds(5)), "the family lock was still held")
        again.cancel()
    }

    /// After app quit begins, the offline cache probe spawns nothing (an install
    /// unwinding from the quit refreshes, which used to start `prefetch --check`).
    @Test func noCacheProbeStartsAfterShutdown() async throws {
        let (manager, paths) = try VariantTests.manager()
        try await VariantTests.fakeEnvironment(.pocketTTS, manager, paths)
        let ran = paths.supportDir.appending(path: "probe-ran")
        try "#!/bin/sh\ntouch '\(ran.path)'\necho '{\"event\":\"summary\",\"cached\":true,\"size_bytes\":1}'\n"
            .write(to: paths.python(.pocketTTS), atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: paths.python(.pocketTTS).path)
        #expect(await manager.modelCacheStatus(.pocketTTS)?.cached == true)
        #expect(FileManager.default.fileExists(atPath: ran.path))
        try FileManager.default.removeItem(at: ran)

        await manager.shutdown(grace: .milliseconds(200), drain: .milliseconds(200))
        #expect(await manager.modelCacheStatus(.pocketTTS) == nil)
        #expect(!FileManager.default.fileExists(atPath: ran.path), "a probe ran after shutdown")
    }
}

// MARK: - Review round 2

/// Frames the real TTS server sent (`tests/test_macos_catalog_drift.py` checks
/// the fixture against a live server on every Python run).
@Suite struct WireFixtureTests {
    static func frames() throws -> [String: String] {
        let url = URL(filePath: #filePath).deletingLastPathComponent().appending(path: "Fixtures/tts_wire_frames.json")
        let object = try #require(try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        return try object.mapValues { String(decoding: try JSONSerialization.data(withJSONObject: $0), as: UTF8.self) }
    }

    @Test func errorFrameCarriesTheServerCodeAndMessage() throws {
        let frames = try Self.frames()
        let busy = try #require(frames["error_busy"])
        let append = try #require(frames["error_append"])
        #expect(
            TTSEvent.decode(busy)
                == .error(code: "busy", message: "synthesis backlog full; retry after backoff", previousEventID: "tryit-2"))
        #expect(
            TTSEvent.decode(append)
                == .error(code: "invalid_config", message: "input_text.append missing text", previousEventID: "tryit-3.append"))
    }

    /// The fixture's exchange as Try it sees it: request 1 plays, is stopped,
    /// and request 2 is refused. Every frame is routed to the request it names.
    @Test func realFramesRouteToTheirRequest() throws {
        let frames = try Self.frames()
        func event(_ name: String) throws -> TTSEvent {
            let frame = try #require(frames[name])
            return try #require(TTSEvent.decode(frame))
        }
        guard case .committed(let rid, _) = try event("committed") else {
            Issue.record("not input_text.committed")
            return
        }
        var t = ResponseTracker()
        #expect(t.committed() == "tryit-1")  // the fixture's ids are the ones the app mints
        #expect(t.route(try event("committed"), speaking: true) == .ignore)
        #expect(t.isCurrent(rid))
        #expect(t.route(try event("created"), speaking: true) == .ignore)
        #expect(t.route(try event("audio"), speaking: true) == .play(Data([0x00, 0x80, 0xFF, 0x7F])))
        t.abandon()  // Stop
        #expect(t.route(try event("audio"), speaking: true) == .ignore)
        #expect(t.route(try event("cancelled"), speaking: true) == .ignore)
        #expect(t.committed() == "tryit-2")
        #expect(t.route(try event("error_busy"), speaking: true) == .fail(ResponseTrackerTests.busy))
        // A refusal naming another request (here a later one) is not this one's.
        #expect(t.route(try event("error_append"), speaking: true) == .ignore)
    }
}

@Suite struct QueuedInstallCancelTests {
    /// Cancel while queued behind a sibling's install: it records nothing,
    /// though the environment it would share exists.
    @Test func aCancelledQueuedInstallRecordsNothing() async throws {
        let (manager, paths) = try VariantTests.manager()
        try await VariantTests.fakeEnvironment(.phonon, manager, paths)
        // `prefetch` blocks, so the first install holds the family lock.
        try "#!/bin/sh\nsleep 30\n".write(to: paths.python(.phonon), atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: paths.python(.phonon).path)

        let first = Task { try await manager.install(.phononQ8).consume { _ in } }
        let deadline = ContinuousClock.now + .seconds(5)
        while !manager.isRequested(.phononQ8), ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(20)) }
        #expect(manager.isRequested(.phononQ8), "the first install never reached its download")

        let queued = Flag()
        let second = Task {
            try await manager.install(.phononQ4).consume { if $0 == .stage(.queued) { queued.raise() } }
        }
        #expect(await queued.wait(.seconds(5)), "the second install was not queued")
        second.cancel()
        first.cancel()
        _ = await first.result
        // The cancelled install unwinds in the background once the lock frees.
        let settle = ContinuousClock.now + .seconds(2)
        while !manager.isRequested(.phononQ4), ContinuousClock.now < settle { try await Task.sleep(for: .milliseconds(20)) }
        if case .failure(let error) = await second.result {
            #expect(error is CancellationError)
        } else {
            Issue.record("the cancelled install finished")
        }
        #expect(!manager.isRequested(.phononQ4), "a cancelled queued install recorded its variant")
    }
    enum Script { case ok, fail, block }
    enum Outcome: Equatable { case finished, failed(InstallStage), cancelled }

    /// Every way a variant install can end, and whether it records the variant:
    /// the marker is written after packages and before the model download, so
    /// only an install that got that far (and was not cancelled first) has one.
    @Test(arguments: [
        // (environment outdated?, uv, prefetch, cancel while running?) -> outcome, recorded
        (false, Script.ok, Script.ok, false, Outcome.finished, true),
        (false, .ok, .fail, false, .failed(.model), true),
        (false, .ok, .block, true, .cancelled, true),  // resumable: needsModel
        (true, .fail, .ok, false, .failed(.packages), false),
        (true, .block, .ok, true, .cancelled, false),
        (true, .ok, .ok, false, .finished, true),
    ])
    func everyEndingRecordsTheVariantOnlyPastPackages(
        outdated: Bool, uv: Script, prefetch: Script, cancel: Bool, outcome: Outcome, recorded: Bool
    ) async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "toolbar-ending-\(UUID().uuidString.prefix(8))")
        let paths = AppPaths(
            supportDir: root, cacheDir: root.appending(path: "cache"), logsDir: root.appending(path: "logs"),
            hubDir: root.appending(path: "hub"), derivedModelsDir: root.appending(path: "derived"))
        try paths.prepare()
        let started = root.appending(path: "started")
        func script(_ kind: Script) -> String {
            switch kind {
            case .ok: "#!/bin/sh\nexit 0\n"
            case .fail: "#!/bin/sh\necho 'error: boom' >&2\nexit 1\n"
            case .block: "#!/bin/sh\ntouch '\(started.path)'\nexec sleep 30\n"
            }
        }
        let uvPath = root.appending(path: "uv")
        try script(uv).write(to: uvPath, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: uvPath.path)
        let manager = EnvironmentManager(
            paths: paths,
            tools: Tools(uv: uvPath, wheel: URL(filePath: "/tmp/pipecat_local_tts_server-0-py3-none-any.whl"), constraints: nil,
                         helper: nil, source: .override),
            cancelGrace: .milliseconds(300))
        try await VariantTests.fakeEnvironment(.phonon, manager, paths)
        if outdated {
            try EnvironmentMarker(service: "phonon", fingerprint: "old.whl", python: "3.12.0", createdAt: Date())
                .write(paths.marker(.phonon))
        }
        try script(prefetch).write(to: paths.python(.phonon), atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: paths.python(.phonon).path)

        let install = Task { try await manager.install(.phononQ8).consume { _ in } }
        if cancel {
            let deadline = ContinuousClock.now + .seconds(5)
            while !FileManager.default.fileExists(atPath: started.path), ContinuousClock.now < deadline {
                try await Task.sleep(for: .milliseconds(20))
            }
            install.cancel()
        }
        let ended: Outcome
        switch await install.result {
        case .success: ended = .finished
        case .failure(let error as InstallError): ended = .failed(error.stage)
        case .failure(let error): ended = error is CancellationError ? .cancelled : .failed(.queued)
        }
        #expect(ended == outcome)
        // A cancelled install unwinds in the background: give it time to (wrongly) record.
        try await Task.sleep(for: .milliseconds(cancel ? 500 : 50))
        #expect(manager.isRequested(.phononQ8) == recorded)
    }
}

@Suite struct PipeDrainTests {
    /// A child that exits while a grandchild still holds its pipes.
    static let leaky = "(sleep 20) &\necho out\necho failed >&2\nexit 3\n"

    @Test func anInstallStepEndsWhenAGrandchildHoldsItsPipes() async throws {
        let paths = try InstallCancellationTests.paths()
        let uv = paths.supportDir.appending(path: "uv")
        try "#!/bin/sh\n\(Self.leaky)".write(to: uv, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: uv.path)
        let manager = EnvironmentManager(
            paths: paths, tools: Tools(uv: uv, wheel: URL(filePath: "/tmp/x.whl"), constraints: nil, helper: nil, source: .override))
        let ended = Flag()
        let install = Task {
            defer { ended.raise() }
            try await manager.install(.tone).consume { _ in }
        }
        #expect(await ended.wait(.seconds(8)), "the install waited for the grandchild")
        if case .failure(let error) = await install.result {
            #expect((error as? InstallError)?.stage == .python)
        } else {
            Issue.record("a failing step finished")
        }
    }

    @Test func collectEndsWhenAGrandchildHoldsItsPipes() async throws {
        let sub = Subprocess(
            .init(executable: URL(filePath: "/bin/sh"), arguments: ["-c", Self.leaky], environment: ProcessEnvironment.base()))
        let ended = Flag()
        let result = Task {
            defer { ended.raise() }
            return try await sub.collect()
        }
        #expect(await ended.wait(.seconds(8)), "collect waited for the grandchild")
        let collected = try await result.value
        #expect(collected.status == 3 && collected.stdout == ["out"] && collected.stderr == ["failed"])
    }
    enum Child: CaseIterable { case exits, fails, leaksAndFails, ignoresTermThenCancelled, leaksThenCancelled }

    /// Every way an install subprocess (a uv step or `prefetch`) can end, with
    /// and without a grandchild holding its pipes: the install ends promptly and
    /// the family lock is free for the next one.
    @Test(arguments: Child.allCases, [false, true])
    func everyChildEndingReleasesTheFamily(child: Child, inPrefetch: Bool) async throws {
        let paths = try InstallCancellationTests.paths()
        let started = paths.supportDir.appending(path: "started")
        let body: String
        switch child {
        case .exits: body = "[ \"$1\" = venv ] && mkdir -p \"$4\"\nexit 0"
        case .fails: body = "exit 3"
        case .leaksAndFails: body = "(sleep 20) &\nexit 3"
        case .ignoresTermThenCancelled: body = "trap '' TERM\ntouch '\(started.path)'\nsleep 20"
        case .leaksThenCancelled: body = "(sleep 20) &\ntouch '\(started.path)'\nexec sleep 20"
        }
        let script = paths.supportDir.appending(path: "child")
        try "#!/bin/sh\n\(body)\n".write(to: script, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)
        let manager = EnvironmentManager(
            paths: paths,
            tools: Tools(
                uv: inPrefetch ? URL(filePath: "/usr/bin/true") : script,
                wheel: URL(filePath: "/tmp/pipecat_local_tts_server-0-py3-none-any.whl"), constraints: nil, helper: nil,
                source: .override),
            cancelGrace: .milliseconds(300), drainLimit: .milliseconds(300))
        let service: Service = inPrefetch ? .phonon : .tone
        if inPrefetch {
            try await VariantTests.fakeEnvironment(.phonon, manager, paths)
            try FileManager.default.removeItem(at: paths.python(.phonon))
            try FileManager.default.copyItem(at: script, to: paths.python(.phonon))
        }

        let ended = Flag()
        let install = Task {
            defer { ended.raise() }
            try await manager.install(service).consume { _ in }
        }
        if child == .ignoresTermThenCancelled || child == .leaksThenCancelled {
            let deadline = ContinuousClock.now + .seconds(5)
            while !FileManager.default.fileExists(atPath: started.path), ContinuousClock.now < deadline {
                try await Task.sleep(for: .milliseconds(20))
            }
            install.cancel()
        }
        #expect(await ended.wait(.seconds(5)), "\(child): the install did not end")
        let result = await install.result
        switch child {
        case .exits: if case .failure(let error) = result { Issue.record("\(child): \(error)") }
        case .fails, .leaksAndFails: #expect((try? result.get()) == nil)
        case .ignoresTermThenCancelled, .leaksThenCancelled:
            if case .failure(let error) = result { #expect(error is CancellationError) } else { Issue.record("finished") }
        }
        // The lock comes free (a cancelled install unwinds in the background): the
        // next install of the family gets past `.queued`.
        let progressed = Flag()
        let next = Task {
            try? await manager.install(service).consume { if case .stage(let s) = $0, s != .queued { progressed.raise() } }
        }
        #expect(await progressed.wait(.seconds(3)), "\(child): the family lock was still held")
        next.cancel()
        _ = await next.value
    }
}

@Suite struct STTServerLifetimeTests {
    /// Stop and release the server while a client is connected: the connection's
    /// late callbacks (cancelled state, close completion) must not touch it.
    @Test func connectionsOutliveAReleasedServer() async throws {
        // A decode in flight keeps its session alive past stop().
        var server: STTWebSocketServer? = STTWebSocketServer(identity: .init(name: "slow", model: nil)) { _, _ in
            // Ignores cancellation, as a real decode can.
            await withCheckedContinuation { done in DispatchQueue.global().asyncAfter(deadline: .now() + 0.6) { done.resume() } }
            return "late"
        }
        let port = try await server!.start(port: 0)
        let client = STTClient(port: port)
        _ = try await client.connect()
        let pending = Task { try? await client.transcribe(Data(count: 3200)) }
        try await Task.sleep(for: .milliseconds(150))
        weak let released = server
        let queue = server!.queue
        let gate = DispatchSemaphore(value: 0)
        server?.stop()
        // Hold the server queue so the connection's callbacks run after the release.
        queue.async { gate.wait() }
        server = nil
        #expect(released == nil, "the test needs the server gone")
        gate.signal()
        try await Task.sleep(for: .milliseconds(1000))
        _ = await pending.value
        await client.close()
    }

    @Test func connectionsBeyondTheCapAreRefused() async throws {
        let server = STTServerTests.fakeServer()
        let port = try await server.start(port: 0)
        defer { server.stop() }
        var clients: [STTClient] = []
        for _ in 0..<STTWebSocketServer.maxConnections {
            let client = STTClient(port: port)
            _ = try await client.connect()
            clients.append(client)
        }
        let extra = STTClient(port: port)
        await #expect(throws: (any Error).self) { _ = try await extra.connect() }
        // A freed slot is usable again.
        await clients.removeFirst().close()
        try await Task.sleep(for: .milliseconds(200))
        let again = STTClient(port: port)
        _ = try await again.connect()
        for client in clients + [again, extra] { await client.close() }
    }

    @Test func decodesRunOneAtATime() async throws {
        let gauge = ConcurrencyGauge()
        let server = STTWebSocketServer(identity: .init(name: "slow", model: nil)) { pcm, _ in
            gauge.enter()
            defer { gauge.leave() }
            try await Task.sleep(for: .milliseconds(200))
            return "heard \(pcm.count)"
        }
        let port = try await server.start(port: 0)
        defer { server.stop() }
        let a = STTClient(port: port)
        let b = STTClient(port: port)
        _ = try await a.connect()
        _ = try await b.connect()
        async let first = a.transcribe(Data(count: 3200))
        async let second = b.transcribe(Data(count: 6400))
        #expect(try await [first, second] == ["heard 3200", "heard 6400"])
        #expect(gauge.peak == 1, "two transcriptions ran at once")
        await a.close()
        await b.close()
    }
}

final class ConcurrencyGauge: @unchecked Sendable {
    private let lock = NSLock()
    private var now = 0
    private var max = 0
    func enter() { lock.withLock { now += 1; max = Swift.max(max, now) } }
    func leave() { lock.withLock { now -= 1 } }
    var peak: Int { lock.withLock { max } }
}

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

        try await client.speak("Hello there.", voice: nil)
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
    @Test func staleCancelAfterStopDoesNotStopTheNextUtterance() {
        var t = ResponseTracker()
        t.committed()
        t.created("r1")
        #expect(t.isCurrent("r1"))
        t.abandon()  // Stop
        t.committed()  // Speak again before r1's cancel acknowledgement
        #expect(!t.isCurrent("r1"))  // the late response.cancelled{r1} is ignored
        t.created("r2")
        #expect(t.isCurrent("r2") && !t.isCurrent("r1"))
    }

    @Test func stopBeforeCreatedSkipsThatResponse() {
        var t = ResponseTracker()
        t.committed()
        t.abandon()
        t.committed()
        t.created("r1")  // the abandoned commit's response
        #expect(!t.isCurrent("r1") && t.current == nil)
        t.created("r2")
        #expect(t.isCurrent("r2"))
    }

    @Test func refusedCommitCreatesNothing() {
        var t = ResponseTracker()
        t.committed()
        t.refused()  // e.g. busy
        t.committed()
        t.created("r2")
        #expect(t.isCurrent("r2"))
        #expect(!t.isCurrent(nil))
    }

    /// Every sequence of up to 7 operations, against a direct model: the
    /// server answers commits in order, and only the newest, un-stopped commit's
    /// response is current.
    @Test func matchesTheCommitOrderModelForEverySequence() {
        enum Op: CaseIterable { case commit, created, refused, abandon }
        func check(_ ops: [Op]) {
            var tracker = ResponseTracker()
            // Model: commits awaiting an answer, by index; which index is wanted.
            var pending: [Int] = []
            var wantedCommit: Int?
            var currentCommit: Int?
            var commits = 0
            var ids: [Int: String] = [:]
            for op in ops {
                switch op {
                case .commit:
                    pending.append(commits)
                    wantedCommit = commits
                    currentCommit = nil
                    commits += 1
                    tracker.committed()
                case .created:
                    guard !pending.isEmpty else { return }  // the server never does this
                    let commit = pending.removeFirst()
                    let id = "r\(commit)"
                    ids[commit] = id
                    if commit == wantedCommit { currentCommit = commit }
                    tracker.created(id)
                case .refused:
                    guard !pending.isEmpty else { return }
                    pending.removeFirst()
                    tracker.refused()
                case .abandon:
                    wantedCommit = nil
                    currentCommit = nil
                    tracker.abandon()
                }
                #expect(tracker.current == currentCommit.flatMap { ids[$0] }, "\(ops)")
                for (commit, id) in ids where commit != currentCommit {
                    #expect(!tracker.isCurrent(id), "\(ops)")
                }
            }
        }
        var sequences: [[Op]] = [[]]
        for _ in 0..<7 {
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

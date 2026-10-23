import Foundation
import Testing

@testable import PipecatToolbarCore

/// The app's real install -> serve -> use -> remove path, through `uv`, the TTS
/// server wheel and the STT helper, into a throwaway support directory. Opt-in
/// because it downloads Python, packages and model weights (phonon ~100 MB,
/// Kyutai ~2.4 GB; weights are shared with the Hugging Face cache):
///
///     PIPECAT_TOOLBAR_E2E=1 swift test --package-path macos --filter EndToEnd
///
/// Uses the built bundle's tools when `macos/build/Pipecat Service Toolbar.app`
/// exists (exactly what the app ships), else builds them from the repo.
@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["PIPECAT_TOOLBAR_E2E"] == "1"))
struct EndToEndTests {
    static let macosDir = URL(filePath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent()

    static func makeManager() async throws -> (EnvironmentManager, AppPaths) {
        let root = FileManager.default.temporaryDirectory.appending(path: "pipecat-toolbar-e2e-\(UUID().uuidString.prefix(8))")
        let real = AppPaths.standard()
        // Fresh support dir; the real uv cache so repeat runs are fast.
        let paths = AppPaths(supportDir: root.appending(path: "support"), cacheDir: real.cacheDir, logsDir: root.appending(path: "logs"))
        try paths.prepare()
        let bundleResources = macosDir.appending(path: "build/Pipecat Service Toolbar.app/Contents/Resources")
        let tools: Tools
        if let bundle = Bundle(url: macosDir.appending(path: "build/Pipecat Service Toolbar.app")),
            FileManager.default.fileExists(atPath: bundleResources.appending(path: "bin/uv").path)
        {
            tools = try await ToolLocator.locate(bundle: bundle, paths: paths)
            #expect(tools.source == .bundle)
        } else {
            tools = try await ToolLocator.locate(paths: paths)
        }
        return (EnvironmentManager(paths: paths, tools: tools), paths)
    }

    func install(_ backend: Service, _ manager: EnvironmentManager) async throws -> [InstallEvent] {
        var events: [InstallEvent] = []
        for try await event in manager.install(backend) { events.append(event) }
        return events
    }

    /// Starts `service` via the manager's supervised serve command on a free port.
    func startServer(_ service: Service, _ manager: EnvironmentManager, paths: AppPaths) async throws -> (Subprocess, Int) {
        let args = try #require(manager.serveArguments(for: service.info, port: 0))
        let server = Subprocess(
            .init(executable: paths.python(service), arguments: args, environment: ProcessEnvironment.base(), holdStdin: true))
        try server.run()
        for await line in server.stdout {
            guard let event = ServerCLIEvent.parse(line) else { continue }
            if event.event == "ready", let port = event.port { return (server, port) }
            if event.event == "error" { break }
        }
        server.kill()
        throw TTSClientError.closed
    }

    func serveAndSpeak(_ backend: Service, paths: AppPaths) async throws -> Int {
        let server = Subprocess(
            .init(
                executable: paths.python(backend),
                arguments: ["-m", "tts_server", "serve", "--backend", backend.rawValue, "--host", "127.0.0.1", "--port", "0", "--supervised"],
                environment: ProcessEnvironment.base(), holdStdin: true))
        try server.run()
        defer { server.kill() }
        var port: Int?
        for await line in server.stdout {
            guard let event = ServerCLIEvent.parse(line) else { continue }
            if event.event == "ready" { port = event.port; break }
            if event.event == "error" { Issue.record("server error: \(event.message ?? "")"); break }
        }
        let client = TTSClient(port: try #require(port))
        let hello = try await client.connect()
        #expect(hello.backend == backend.rawValue)
        try await client.speak("Testing the menu bar app.", voice: nil)
        var bytes = 0
        while let event = try await client.next() {
            if case .audio(_, _, let pcm) = event { bytes += pcm.count }
            if case .done = event { break }
            if case .failed(_, let message) = event { Issue.record("failed: \(message)"); break }
        }
        await client.close()
        server.closeStdin()
        #expect(await server.waitForExit() == 0)
        return bytes
    }

    @Test func toneInstallServeRemove() async throws {
        let (manager, paths) = try await Self.makeManager()
        let events = try await install(.tone, manager)
        #expect(events.contains(.stage(.packages)))
        #expect(!events.contains(.stage(.model)))
        guard case .ready(let marker) = manager.environmentStatus(.tone) else {
            Issue.record("tone env not ready")
            return
        }
        #expect(marker.python.hasPrefix("3.12"))
        #expect(try await serveAndSpeak(.tone, paths: paths) > 0)
        try await manager.remove(.tone)
        #expect(manager.environmentStatus(.tone) == .missing)
    }

    @Test func phononInstallDownloadServe() async throws {
        let (manager, paths) = try await Self.makeManager()
        let events = try await install(.phonon, manager)
        #expect(events.contains(.stage(.model)))
        let cache = try #require(await manager.modelCacheStatus(.phonon))
        #expect(cache.cached && cache.sizeBytes > 50_000_000)
        #expect(cache.repoFolders.first?.lastPathComponent == "models--gradium--phonon-31m-en")
        // 24 kHz pcm16: a short sentence is well over 1 s of audio.
        #expect(try await serveAndSpeak(.phonon, paths: paths) > 48_000)
        try await manager.remove(.phonon)  // keeps the shared HF weights
    }

    /// A spoken clip from the system voice, as pcm16 16 kHz mono.
    static func spokenClip(_ text: String) throws -> Data {
        let dir = FileManager.default.temporaryDirectory.appending(path: "toolbar-clip-\(UUID().uuidString.prefix(6))")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let aiff = dir.appending(path: "clip.aiff"), wav = dir.appending(path: "clip.wav")
        for (tool, args) in [("/usr/bin/say", ["-o", aiff.path, text]), ("/usr/bin/afconvert", ["-f", "WAVE", "-d", "LEI16@16000", "-c", "1", aiff.path, wav.path])] {
            let p = Process()
            p.executableURL = URL(filePath: tool)
            p.arguments = args
            try p.run()
            p.waitUntilExit()
        }
        let data = try Data(contentsOf: wav)
        return data.subdata(in: 44..<data.count)  // canonical 44-byte WAV header
    }

    @Test func kyutaiInstallServeTranscribe() async throws {
        let (manager, paths) = try await Self.makeManager()
        let events = try await install(.kyutai, manager)
        #expect(events.contains(.stage(.model)))
        let cache = try #require(await manager.modelCacheStatus(.kyutai))
        #expect(cache.cached && cache.sizeBytes > 2_000_000_000)
        #expect(cache.repoFolders.first?.lastPathComponent == "models--kyutai--stt-1b-en_fr-mlx")
        let (server, port) = try await startServer(.kyutai, manager, paths: paths)
        defer { server.kill() }
        let client = STTClient(port: port)
        #expect(try await client.connect().backend == "kyutai")
        let text = try await client.transcribe(try Self.spokenClip("The weather today is sunny and warm."))
        #expect(text.lowercased().contains("weather"))
        await client.close()
        server.closeStdin()
        #expect(await server.waitForExit() == 0)
        try await manager.remove(.kyutai)
    }
}

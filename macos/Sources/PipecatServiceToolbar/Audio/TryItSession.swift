import Foundation
import Observation
import PipecatToolbarCore

/// "Try it": sends text to a running server and plays the streamed audio.
@MainActor
@Observable
final class TryItSession {
    enum Phase: Equatable {
        case idle
        case connecting
        case speaking
        case failed(String)
    }

    var text: String
    var voices: [String] = []
    /// `nil` = the server's default voice.
    var voice: String?
    private(set) var phase: Phase = .idle
    private(set) var hello: ServerHello?

    let backend: Service
    private let port: Int
    private var client: TTSClient?
    /// Orders `client`'s request frames (clear, append, commit) and Stop's cancel.
    private var requests: TTSRequestQueue?
    /// The connect in flight, shared so `prepare()` and `speak()` open one client.
    private var connecting: Task<TTSClient, Error>?
    private var receiveTask: Task<Void, Never>?
    private var responses = ResponseTracker()
    private let player = AudioPlayer()

    init(backend: Service, port: Int) {
        self.backend = backend
        self.port = port
        self.text = UserDefaults.standard.string(forKey: Self.textKey) ?? Self.sample(for: backend)
        self.voice = UserDefaults.standard.string(forKey: voiceKey)
        player.onDrained = { [weak self] in
            guard let self, self.phase == .speaking else { return }
            self.phase = .idle
        }
        player.onInterrupted = { [weak self] in
            guard let self, self.phase == .speaking else { return }
            // Like Stop: the rest of this response is not wanted.
            self.responses.abandon()
            self.requests?.cancel()
            self.phase = .failed("Playback stopped: the audio output changed. Press Speak to try again.")
        }
    }

    private static let textKey = "tryIt.text"
    private var voiceKey: String { "tryIt.voice.\(backend.rawValue)" }

    static func sample(for backend: Service) -> String {
        switch backend {
        case .dia: "[S1] Welcome to Pipecat. [S2] It all runs on this Mac."
        case .tone: "A short test tone."
        default: "Hello from Pipecat. This voice is running entirely on your Mac."
        }
    }

    var isBusy: Bool { phase == .connecting || phase == .speaking }
    var isSpeaking: Bool { phase == .speaking }
    var overLimit: Bool {
        guard let limit = hello?.maxTextChars else { return false }
        return text.count > limit
    }

    func speak() async {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !isBusy else { return }
        UserDefaults.standard.set(text, forKey: Self.textKey)
        if let voice { UserDefaults.standard.set(voice, forKey: voiceKey) }
        do {
            phase = .connecting
            _ = try await connectedClient()
            // Stop (or detach) while connecting: this request is not wanted.
            guard phase == .connecting, let hello, let requests else { return }
            try player.begin(rate: hello.rate)
            phase = .speaking
            let request = responses.committed()
            try await requests.speak(trimmed, voice: voice, eventID: request)
        } catch {
            // A request superseded by Stop does not report its failure.
            if phase == .speaking || phase == .connecting { fail(error.localizedDescription) }
        }
    }

    func stop() async {
        player.stop()
        responses.abandon()
        // Queued behind the request's frames: the cancel follows its commit.
        if phase == .speaking { requests?.cancel() }
        phase = .idle
    }

    /// Loads voices without speaking (when the detail view opens). A failure here
    /// stays silent; pressing Speak retries and reports it.
    func prepare() async {
        guard client == nil else { return }
        _ = try? await connectedClient()
    }

    func detach() {
        connecting?.cancel()
        connecting = nil
        receiveTask?.cancel()
        receiveTask = nil
        responses = ResponseTracker()
        if let client { Task { await client.close() } }
        client = nil
        requests = nil
        player.stop()
        if phase != .idle { phase = .idle }
    }

    private func connectedClient() async throws -> TTSClient {
        if let client { return client }
        if let connecting { return try await connecting.value }
        let connect = Task { () throws -> TTSClient in
            let client = TTSClient(port: port)
            let hello: ServerHello
            do {
                hello = try await client.connect()
                try Task.checkCancellation()
            } catch {
                await client.close()
                throw error
            }
            self.client = client
            self.requests = TTSRequestQueue { frame in try await client.send(frame) }
            self.hello = hello
            receiveTask = Task { [weak self] in await self?.receive(from: client) }
            if hello.voiceCount > 0 { try await client.requestStatus() }
            return client
        }
        connecting = connect
        defer { if connecting == connect { connecting = nil } }
        return try await connect.value
    }

    private func receive(from client: TTSClient) async {
        do {
            while let event = try await client.next() {
                handle(event)
            }
        } catch {
            if !Task.isCancelled, self.client === client {
                self.client = nil
                self.requests = nil
                if phase == .speaking { fail("The server closed the connection.") }
            }
        }
    }

    private func handle(_ event: TTSEvent) {
        switch responses.route(event, speaking: phase == .speaking) {
        case .ignore:
            break
        case .voices(let voices):
            self.voices = voices
            if let voice, !voices.contains(voice) { self.voice = nil }
        case .play(let pcm):
            player.enqueue(pcm16: pcm)
        case .finish:
            player.finish()
        case .stopped:
            player.stop()
            if phase == .speaking { phase = .idle }
        case .fail(let message):
            fail(message)
        }
    }

    private func fail(_ message: String) {
        player.stop()
        phase = .failed(message)
    }
}

import Foundation

/// `server.hello`: what the running server advertises.
public struct ServerHello: Sendable, Equatable {
    public var backend: String
    public var model: String?
    public var rate: Int
    public var channels: Int
    public var streaming: Bool
    public var voiceCount: Int
    public var maxTextChars: Int?
    public var languages: [String]
}

/// A server -> client event, decoded from one JSON text frame (docs/protocol.md §5).
public enum TTSEvent: Sendable, Equatable {
    case hello(ServerHello)
    case status(voices: [String], queueDepth: Int?)
    /// `input_text.committed`: the commit whose `event_id` was `previousEventID` became `responseID`.
    case committed(responseID: String, previousEventID: String?)
    case responseCreated(id: String)
    case audio(responseID: String, seq: Int, pcm: Data)
    case done(responseID: String, durationMs: Int?)
    case cancelled(responseID: String)
    case failed(responseID: String?, message: String)
    /// `previousEventID` is the `event_id` of the client frame the server refused, if it had one.
    case error(code: String, message: String, previousEventID: String?)
    case other(type: String)

    public static func decode(_ text: String) -> TTSEvent? {
        guard let data = text.data(using: .utf8),
            let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let type = obj["type"] as? String
        else { return nil }
        let rid = obj["response_id"] as? String
        switch type {
        case "server.hello":
            let audio = obj["audio"] as? [String: Any] ?? [:]
            let caps = obj["capabilities"] as? [String: Any] ?? [:]
            let backend = obj["backend"] as? [String: Any] ?? [:]
            return .hello(
                ServerHello(
                    backend: backend["name"] as? String ?? "",
                    model: backend["model"] as? String,
                    rate: audio["rate"] as? Int ?? 24000,
                    channels: audio["channels"] as? Int ?? 1,
                    streaming: caps["streaming"] as? Bool ?? false,
                    voiceCount: caps["voice_count"] as? Int ?? 0,
                    maxTextChars: caps["max_text_chars"] as? Int,
                    languages: caps["languages"] as? [String] ?? []))
        case "server.status":
            return .status(voices: obj["voices"] as? [String] ?? [], queueDepth: obj["queue_depth"] as? Int)
        case "input_text.committed":
            return .committed(responseID: rid ?? "", previousEventID: obj["previous_event_id"] as? String)
        case "response.created":
            return .responseCreated(id: rid ?? "")
        case "response.audio.delta":
            guard let b64 = obj["audio"] as? String, let pcm = Data(base64Encoded: b64) else {
                return .other(type: type)
            }
            return .audio(responseID: rid ?? "", seq: obj["seq"] as? Int ?? 0, pcm: pcm)
        case "response.audio.done":
            return .done(responseID: rid ?? "", durationMs: obj["duration_ms"] as? Int)
        case "response.cancelled":
            return .cancelled(responseID: rid ?? "")
        case "response.failed":
            let err = obj["error"] as? [String: Any]
            return .failed(responseID: rid, message: err?["message"] as? String ?? "Synthesis failed")
        case "error":
            // Nested, OpenAI-Realtime style: {type: "error", error: {type, code, message}} (docs/protocol.md §7).
            let err = obj["error"] as? [String: Any] ?? [:]
            return .error(
                code: err["code"] as? String ?? "error", message: err["message"] as? String ?? "",
                previousEventID: obj["previous_event_id"] as? String)
        default:
            return .other(type: type)
        }
    }
}

public enum TTSClientError: LocalizedError, Sendable {
    case unexpectedHandshake
    case closed

    public var errorDescription: String? {
        switch self {
        case .unexpectedHandshake: "The server did not send server.hello."
        case .closed: "The connection closed."
        }
    }
}

/// A minimal client for the local TTS server's WebSocket protocol.
///
/// `URLSessionWebSocketTask` sends no `Origin` header (the server rejects any
/// Origin with 403). One consumer calls `next()` in a loop while other tasks
/// call the send methods; actor reentrancy lets sends run while `next()` waits.
public actor TTSClient {
    private let session: URLSession
    private let task: URLSessionWebSocketTask

    public init(host: String = ServiceCatalog.host, port: Int) {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 600
        session = URLSession(configuration: config)
        var request = URLRequest(url: URL(string: "ws://\(host):\(port)/")!)
        request.timeoutInterval = 10
        task = session.webSocketTask(with: request)
        task.maximumMessageSize = 16 * 1024 * 1024
    }

    /// Opens the socket and waits for `server.hello`.
    public func connect() async throws -> ServerHello {
        task.resume()
        guard case .hello(let hello)? = try await next() else {
            throw TTSClientError.unexpectedHandshake
        }
        return hello
    }

    /// The next decodable server event (unknown frames are skipped).
    public func next() async throws -> TTSEvent? {
        while true {
            let message = try await task.receive()
            let text: String?
            switch message {
            case .string(let s): text = s
            case .data(let d): text = String(data: d, encoding: .utf8)
            @unknown default: text = nil
            }
            if let text, let event = TTSEvent.decode(text) { return event }
        }
    }

    public func requestStatus() async throws {
        try await send(["type": "server.status"])
    }

    public func close() {
        task.cancel(with: .normalClosure, reason: nil)
        session.invalidateAndCancel()
    }

    /// Sends one client frame. Try it's requests go through `TTSRequestQueue`,
    /// which owns their order.
    public func send(_ object: [String: String]) async throws {
        let data = try JSONSerialization.data(withJSONObject: object)
        try await task.send(.string(String(decoding: data, as: UTF8.self)))
    }
}

/// Puts one connection's Try it frames on the wire in the order they were
/// asked for, so the request lifecycle (docs/protocol.md §4) holds:
///
/// - Every request is `input_text.clear`, `input_text.append`,
///   `input_text.commit`. The server keeps the buffer on any refusal (busy,
///   invalid voice, a too-long append, ...) so a client can retry; Try it never
///   retries, so the clear drops that text before it can prefix the next request.
/// - `cancel()` (Stop) goes on the wire after every frame of a request asked
///   for before it, so its `response.cancel` follows the commit it stops: a
///   cancel that overtook its commit would be a no-op and the commit would
///   still be admitted, holding the connection's single in-flight slot (K=1).
/// - A request stopped (or superseded) before all its frames were sent sends no
///   more of them, so a stopped request's commit is never sent late.
///
/// The order is the order of the calls on the main actor: each call queues its
/// frames before it first suspends.
@MainActor
public final class TTSRequestQueue {
    public typealias Send = @Sendable ([String: String]) async throws -> Void

    private let send: Send
    /// The last queued job; each waits for the one before it.
    private var tail: Task<Void, Never>?
    /// The request whose remaining frames are still wanted.
    private var wanted: String?

    public init(send: @escaping Send) { self.send = send }

    /// The frames of request `eventID`. The commit carries `eventID` and the
    /// append `ResponseTracker.appendID(eventID)`, so the server's replies (and
    /// refusals) name the request they answer.
    public nonisolated static func frames(_ text: String, voice: String?, eventID: String) -> [[String: String]] {
        var commit: [String: String] = ["type": "input_text.commit", "event_id": eventID]
        if let voice { commit["voice"] = voice }
        return [
            ["type": "input_text.clear"],
            ["type": "input_text.append", "text": text, "event_id": ResponseTracker.appendID(eventID)],
            commit,
        ]
    }

    /// Sends request `eventID` once the frames queued before it are out. Throws
    /// if a frame could not be sent; returns early, sending nothing more, once
    /// the request is stopped or another one is asked for.
    public func speak(_ text: String, voice: String?, eventID: String) async throws {
        wanted = eventID
        let frames = Self.frames(text, voice: voice, eventID: eventID)
        try await schedule { [send] queue in
            for frame in frames {
                guard queue.wanted == eventID else { return }
                try await send(frame)
            }
        }.value
    }

    /// Stop: no frame of an earlier request is sent from now on, and a
    /// `response.cancel` follows the ones already sent.
    public func cancel() {
        wanted = nil
        schedule { [send] _ in try await send(["type": "response.cancel"]) }
    }

    /// Returns once every job queued so far has run.
    public func drained() async {
        await tail?.value
    }

    /// Queues `job` behind every job queued before it (synchronously: the
    /// queue order is the call order).
    @discardableResult
    private func schedule(_ job: @escaping @MainActor @Sendable (TTSRequestQueue) async throws -> Void) -> Task<Void, Error> {
        let previous = tail
        let run = Task { @MainActor in
            await previous?.value
            try await job(self)
        }
        tail = Task { _ = await run.result }
        return run
    }
}

/// Which server events on one connection belong to the request the user is
/// waiting for, so events of an abandoned request (a late `response.cancelled`,
/// audio after Stop, a refusal of a stopped commit) never act on the next one.
///
/// Each request's frames carry an `event_id` (`committed()` mints it). The
/// server echoes it as `previous_event_id` on `input_text.committed`, which
/// names the request's `response_id`, and on any `error` refusing one of its
/// frames. Ownership is by those ids, never by arrival order.
public struct ResponseTracker: Sendable, Equatable {
    public enum Owner: Sendable, Equatable {
        /// Answers the newest, un-stopped request.
        case current
        /// Answers an older or stopped request.
        case stale
        /// Names no request (the server could not correlate it).
        case unknown
    }

    private var serial = 0
    /// The event id of the newest request, while it is still wanted.
    private var request: String?
    public private(set) var current: String?

    public init() {}

    /// A request is about to be sent: it is now the one that counts. Returns the
    /// `event_id` its commit carries.
    public mutating func committed() -> String {
        serial += 1
        let id = "tryit-\(serial)"
        request = id
        current = nil
        return id
    }

    /// The user stopped: nothing in flight counts any more.
    public mutating func abandon() {
        request = nil
        current = nil
    }

    /// The `event_id` the append of request `id` carries.
    public static func appendID(_ id: String) -> String { id + ".append" }

    public func owner(of previousEventID: String?) -> Owner {
        guard let previousEventID else { return .unknown }
        guard let request, previousEventID == request || previousEventID == Self.appendID(request) else { return .stale }
        return .current
    }

    public func isCurrent(_ responseID: String?) -> Bool { responseID != nil && responseID == current }

    /// What `event` means for the request in flight. `speaking`: its audio is
    /// still wanted (Try it is playing it).
    public mutating func route(_ event: TTSEvent, speaking: Bool) -> TryItAction {
        switch event {
        case .status(let voices, _):
            return .voices(voices)
        case .committed(let rid, let previous):
            if owner(of: previous) == .current { current = rid }
            return .ignore
        case .audio(let rid, _, let pcm):
            return speaking && isCurrent(rid) ? .play(pcm) : .ignore
        case .done(let rid, _):
            return speaking && isCurrent(rid) ? .finish : .ignore
        case .cancelled(let rid):
            return isCurrent(rid) ? .stopped : .ignore
        case .failed(let rid, let message):
            if isCurrent(rid) || (rid == nil && speaking) { return .fail(message) }
            return .ignore
        case .error(let code, let message, let previous):
            switch owner(of: previous) {
            case .current: break
            case .stale: return .ignore
            // Uncorrelated (the server's own failure): it can only hurt the request playing.
            case .unknown: guard speaking else { return .ignore }
            }
            return .fail(code == "busy" ? "The server is busy. Try again in a moment." : message)
        case .hello, .responseCreated, .other:
            return .ignore
        }
    }
}

/// What Try it does with one server event (`ResponseTracker.route`).
public enum TryItAction: Sendable, Equatable {
    case ignore
    case voices([String])
    case play(Data)
    /// The response is complete: play out what is queued.
    case finish
    /// The server cancelled the response.
    case stopped
    case fail(String)
}

/// Little-endian pcm16 -> Float32 in [-1, 1), as the server documents its frames.
public enum PCM16 {
    public static func floats(from data: Data) -> [Float] {
        let count = data.count / 2
        var out = [Float](repeating: 0, count: count)
        data.withUnsafeBytes { raw in
            for i in 0..<count {
                let lo = UInt16(raw[2 * i])
                let hi = UInt16(raw[2 * i + 1])
                out[i] = Float(Int16(bitPattern: lo | (hi << 8))) / 32768
            }
        }
        return out
    }
}

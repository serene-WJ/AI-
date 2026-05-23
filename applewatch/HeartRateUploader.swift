import Foundation

struct HeartRateUploader {
    private let endpoint: URL
    private let encoder: JSONEncoder

    init(baseURL: URL = AppConfig.backendBaseURL) {
        self.endpoint = baseURL.appendingPathComponent("ingest/watch")
        self.encoder = JSONEncoder()
        self.encoder.dateEncodingStrategy = .iso8601
    }

    func upload(_ payload: HeartRatePayload) async throws {
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try encoder.encode(payload)

        let (_, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse,
              (200..<300).contains(httpResponse.statusCode) else {
            throw URLError(.badServerResponse)
        }
    }
}

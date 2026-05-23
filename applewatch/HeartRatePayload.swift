import Foundation

struct HeartRatePayload: Codable {
    let source: String
    let heartRate: Double
    let unit: String
    let timestamp: Date
}


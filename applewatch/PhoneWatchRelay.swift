import Foundation
import WatchConnectivity

@MainActor
final class PhoneWatchRelay: NSObject, ObservableObject {
    @Published var lastHeartRate: Double?
    @Published var status = "正在等待 Watch 连接"

    private let uploader = HeartRateUploader()

    override init() {
        super.init()
        activateWatchConnectivity()
    }

    private func activateWatchConnectivity() {
        guard WCSession.isSupported() else {
            status = "当前设备不支持 WatchConnectivity"
            return
        }

        let session = WCSession.default
        session.delegate = self
        session.activate()
    }

    private func handle(message: [String: Any]) {
        guard let bpm = message["heartRate"] as? Double else {
            status = "收到 Watch 消息，但缺少 heartRate"
            return
        }

        let timestamp = (message["timestamp"] as? TimeInterval).map(Date.init(timeIntervalSince1970:)) ?? Date()
        let payload = HeartRatePayload(
            source: "apple_watch",
            heartRate: bpm,
            unit: "count/min",
            timestamp: timestamp
        )

        lastHeartRate = bpm
        status = "收到 \(Int(bpm.rounded())) BPM，正在上传"

        Task {
            do {
                try await uploader.upload(payload)
                await MainActor.run {
                    self.status = "已上传 \(Int(bpm.rounded())) BPM"
                }
            } catch {
                await MainActor.run {
                    self.status = "上传失败：\(error.localizedDescription)"
                }
            }
        }
    }
}

extension PhoneWatchRelay: WCSessionDelegate {
    nonisolated func session(
        _ session: WCSession,
        activationDidCompleteWith activationState: WCSessionActivationState,
        error: Error?
    ) {
        Task { @MainActor in
            if let error {
                self.status = "WatchConnectivity 激活失败：\(error.localizedDescription)"
            } else {
                self.status = "WatchConnectivity 已激活：\(activationState.rawValue)"
            }
        }
    }

    nonisolated func sessionDidBecomeInactive(_ session: WCSession) {}

    nonisolated func sessionDidDeactivate(_ session: WCSession) {
        WCSession.default.activate()
    }

    nonisolated func session(_ session: WCSession, didReceiveMessage message: [String: Any]) {
        Task { @MainActor in
            self.handle(message: message)
        }
    }

    nonisolated func session(_ session: WCSession, didReceiveUserInfo userInfo: [String: Any] = [:]) {
        Task { @MainActor in
            self.handle(message: userInfo)
        }
    }
}


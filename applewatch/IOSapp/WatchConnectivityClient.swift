import Foundation
import WatchConnectivity

final class WatchConnectivityClient: NSObject {
    static let shared = WatchConnectivityClient()

    private override init() {
        super.init()
        activate()
    }

    private func activate() {
        guard WCSession.isSupported() else { return }
        WCSession.default.delegate = self
        WCSession.default.activate()
    }

    func sendHeartRate(_ bpm: Double, timestamp: Date) {
        guard WCSession.isSupported() else { return }

        let message: [String: Any] = [
            "heartRate": bpm,
            "timestamp": timestamp.timeIntervalSince1970
        ]

        if WCSession.default.isReachable {
            WCSession.default.sendMessage(message, replyHandler: nil) { _ in
                WCSession.default.transferUserInfo(message)
            }
        } else {
            WCSession.default.transferUserInfo(message)
        }
    }
}

extension WatchConnectivityClient: WCSessionDelegate {
    func session(
        _ session: WCSession,
        activationDidCompleteWith activationState: WCSessionActivationState,
        error: Error?
    ) {}
}


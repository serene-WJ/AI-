import SwiftUI

@main
struct WatchHeartCollectorWatchApp: App {
    @StateObject private var heartRateManager = HeartRateManager()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(heartRateManager)
        }
    }
}


import SwiftUI

@main
struct WatchHeartCollectorApp: App {
    @StateObject private var relay = PhoneWatchRelay()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(relay)
        }
    }
}


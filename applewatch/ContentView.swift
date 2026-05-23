import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var relay: PhoneWatchRelay

    var body: some View {
        VStack(spacing: 16) {
            Text("Watch Heart Collector")
                .font(.title2.weight(.semibold))

            if let bpm = relay.lastHeartRate {
                Text("\(Int(bpm.rounded())) BPM")
                    .font(.system(size: 48, weight: .bold, design: .rounded))
            } else {
                Text("等待 Watch 心率数据")
                    .foregroundStyle(.secondary)
            }

            Text(relay.status)
                .font(.footnote)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding()
    }
}


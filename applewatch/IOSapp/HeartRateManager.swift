import Foundation
import HealthKit

@MainActor
final class HeartRateManager: ObservableObject {
    @Published var lastHeartRate: Double?
    @Published var isRunning = false
    @Published var status = "未开始"

    private let healthStore = HKHealthStore()
    private let heartRateType = HKQuantityType.quantityType(forIdentifier: .heartRate)!
    private let pollInterval: TimeInterval = 2.0
    private var timer: Timer?

    func requestPermissionAndStart() {
        guard HKHealthStore.isHealthDataAvailable() else {
            status = "当前设备不可用 HealthKit"
            return
        }

        healthStore.requestAuthorization(toShare: [], read: [heartRateType]) { [weak self] success, error in
            Task { @MainActor in
                guard let self else { return }

                if let error {
                    self.status = "HealthKit 授权失败：\(error.localizedDescription)"
                    return
                }

                guard success else {
                    self.status = "HealthKit 未授权"
                    return
                }

                self.start()
            }
        }
    }

    func start() {
        stop()
        isRunning = true
        status = "正在读取最近一次心率"

        queryLatestHeartRate()
        timer = Timer.scheduledTimer(withTimeInterval: pollInterval, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.queryLatestHeartRate()
            }
        }
    }

    func stop() {
        timer?.invalidate()
        timer = nil
        isRunning = false
        status = "已停止"
    }

    private func queryLatestHeartRate() {
        let sort = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)
        let query = HKSampleQuery(
            sampleType: heartRateType,
            predicate: nil,
            limit: 1,
            sortDescriptors: [sort]
        ) { [weak self] _, samples, error in
            Task { @MainActor in
                guard let self else { return }

                if let error {
                    self.status = "读取失败：\(error.localizedDescription)"
                    return
                }

                guard let sample = samples?.first as? HKQuantitySample else {
                    self.status = "暂无心率样本"
                    return
                }

                let unit = HKUnit.count().unitDivided(by: .minute())
                let bpm = sample.quantity.doubleValue(for: unit)
                self.lastHeartRate = bpm
                self.status = "已读取 \(Int(bpm.rounded())) BPM"
                WatchConnectivityClient.shared.sendHeartRate(bpm, timestamp: sample.endDate)
            }
        }

        healthStore.execute(query)
    }
}

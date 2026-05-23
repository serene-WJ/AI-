[README.md](https://github.com/user-attachments/files/28176917/README.md)
# Watch HeartKit Collector

一个最小 iOS + watchOS 采集样例：

- Watch 端请求 HealthKit 心率读取权限
- Watch 端每 2 秒读取最近一次 `HKQuantityTypeIdentifier.heartRate`
- 通过 WatchConnectivity 发给 iPhone
- iPhone 端 `POST /ingest/watch` 上传给后端

> 说明：HealthKit 的 `heartRate` 不是“强制 1 秒一条”的实时传感器流。系统只有在 Apple Watch 产生新心率样本时才会有新值。若需要稳定的 1~3 秒心率流，通常要在 Watch 端启动 workout session，让系统持续采样。

## Xcode 接入

1. 新建一个 iOS App，并勾选 Apple Watch App。
2. Watch target 开启 HealthKit capability。
3. iOS target 和 Watch target 都链接 WatchConnectivity framework。
4. 将本目录下 `iOSApp` 文件放入 iOS target。
5. 将本目录下 `WatchApp` 文件放入 Watch App target。
6. 在 Watch target 的 `Info.plist` 添加：

```xml
<key>NSHealthShareUsageDescription</key>
<string>用于读取 Apple Watch 最近一次心率数据。</string>
```

7. 修改 `iOSApp/AppConfig.swift` 中的 `backendBaseURL`。

如果后端是本机开发服务，不要在真机上使用 `127.0.0.1`，请改成电脑的局域网 IP，例如 `http://192.168.1.20:8000`。如果使用 HTTP 调试，还需要在 iOS target 的 `Info.plist` 临时允许 ATS：

```xml
<key>NSAppTransportSecurity</key>
<dict>
  <key>NSAllowsArbitraryLoads</key>
  <true/>
</dict>
```

生产环境建议使用 HTTPS。

## 后端请求

iPhone 会发送：

```http
POST /ingest/watch
Content-Type: application/json
```

JSON 示例：

```json
{
  "source": "apple_watch",
  "heartRate": 82.0,
  "unit": "count/min",
  "timestamp": "2026-05-23T13:20:00Z"
}
```

## 运行方式

先运行 iOS App，再运行 Watch App。Watch App 点击授权后会开始轮询最近一次心率；iPhone App 收到后立即上传。

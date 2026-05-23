import Foundation

enum AppConfig {
    // TODO: 改成 iPhone 真机可访问的后端地址，例如：
    // URL(string: "https://api.example.com")!
    // URL(string: "http://192.168.1.20:8000")!
    // 注意：在 iPhone 上 127.0.0.1 指的是手机自己，不是你的电脑。
    static let backendBaseURL = URL(string: "http://127.0.0.1:8000")!
}

import MLX

public enum Sideros {
    public static func gpuIsAvailable() -> Bool {
        Device.gpu.deviceType == .gpu
    }
}

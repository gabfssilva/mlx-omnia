/// One task-shaped inference model: prompt → tokens, audio → segments, prompt → images.
/// The stream is the primitive: autoregressive tasks append elements, iterative ones
/// (diffusion previews) refine them, single-output tasks emit one element and finish.
public protocol Model<Input, Output, Options> {
    associatedtype Input
    associatedtype Output
    associatedtype Options

    func callAsFunction(_ input: Input, options: Options) -> AsyncThrowingStream<Output, Error>
}

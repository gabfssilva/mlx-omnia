import MLX

precedencegroup LayerComposition {
    associativity: left
    lowerThan: MultiplicationPrecedence
}

infix operator >>>: LayerComposition

/// A layer maps a hidden state and a context to a new hidden state and a new context.
/// Each architecture defines its own Context — masks, KV caches, rotary offset — and the
/// combinators only thread it through. Nothing here mutates anything.
typealias Layer<Context> = (MLXArray, Context) -> (MLXArray, Context)

func >>> <Context>(
    lhs: @escaping Layer<Context>, rhs: @escaping Layer<Context>
) -> Layer<Context> {
    { x, context in
        let (y, next) = lhs(x, context)
        return rhs(y, next)
    }
}

/// Both branches see the same input; the gate multiplies the value. The MLPs are this.
func * <Context>(
    lhs: @escaping Layer<Context>, rhs: @escaping Layer<Context>
) -> Layer<Context> {
    { x, context in
        let (gate, afterGate) = lhs(x, context)
        let (up, afterUp) = rhs(x, afterGate)
        return (gate * up, afterUp)
    }
}

/// The residual bifurcation: keep `x`, add the branch back. Deliberately not an
/// operator — hiding the fork inside `>>>` would misstate the graph.
func residual<Context>(_ branch: () -> Layer<Context>) -> Layer<Context> {
    let f = branch()
    return { x, context in
        let (y, next) = f(x, context)
        return (x + y, next)
    }
}

/// A layer that reads the context but does not write to it: the shape of every primitive
/// that carries only weights (norms, projections, rotations, activations).
func stateless<Context>(_ f: @escaping (MLXArray, Context) -> MLXArray) -> Layer<Context> {
    { x, context in (f(x, context), context) }
}

/// A chain is the context-free stratum of the DSL: `(MLXArray) -> MLXArray`. Chains
/// compose with the same operators, mix freely with layers (the result is a layer), and
/// are what `graph` compiles — a layer that reads the context cannot get into a
/// compiled block by construction, only by type error.
typealias Chain = (MLXArray) -> MLXArray

func >>> (lhs: @escaping Chain, rhs: @escaping Chain) -> Chain {
    { x in rhs(lhs(x)) }
}

func >>> <Context>(lhs: @escaping Chain, rhs: @escaping Layer<Context>) -> Layer<Context> {
    { x, context in rhs(lhs(x), context) }
}

func >>> <Context>(lhs: @escaping Layer<Context>, rhs: @escaping Chain) -> Layer<Context> {
    { x, context in
        let (y, next) = lhs(x, context)
        return (rhs(y), next)
    }
}

func * (lhs: @escaping Chain, rhs: @escaping Chain) -> Chain {
    { x in lhs(x) * rhs(x) }
}

/// The scope of a subnetwork, and the compilation contract, decided by type: a pure
/// chain is traced once per input shape and runs fused after — always, nothing pure
/// misses fusion by oversight. A block that reads the context comes out composed and
/// uncompiled, because it cannot be traced: the cache changes shape every step, and the
/// RoPE offset is an op attribute the trace would freeze at its first value.
/// Inside a traced graph, call only raw ops — an MLXNN activation is itself a compiled
/// function with its own lock, and entering it while the trace holds the global eval
/// lock deadlocks against direct callers.
func graph(_ body: () -> Chain) -> Chain {
    compile(body())
}

func graph<Context>(_ body: () -> Layer<Context>) -> Layer<Context> {
    body()
}

// One decode step, drawn.
//
// The daemon traces the graph by running a token through the tree it builds and recording
// which module handed its array to which, so what is drawn here is what the step did — not
// what the config implies. Two mixers side by side are side by side because both were seen
// reading the array the norm wrote, and an edge the trace could not attribute falls back to
// the order the two ran in and is drawn dashed rather than passed off as a reading.
//
// Two levels, drawn two ways so a reader never has to be told which is which. The trunk runs
// across, once, and stands for the whole model; the block runs down, and stands for one of
// the layers the trunk repeats. A hybrid has more than one kind of block and the range chips
// say which layers run the one on screen.
//
// The layout is a layered walk: a node's row is the longest path to it from the block's
// input, which puts everything that reads the same array on the same row. What skips rows —
// a residual reaching past four of them to its sum — leaves the column and comes down the
// left gutter, which is the shape it has in the code.

import SwiftUI

extension CGPoint {
    func shifted(by direction: CGPoint, _ by: CGFloat) -> CGPoint {
        CGPoint(x: x + direction.x * by, y: y + direction.y * by)
    }
}

struct AnatomyView: View {
    @Environment(\.tokens) private var t
    let blueprint: Blueprint?
    let trouble: String

    @State private var opened: Int?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            if let blueprint {
                Trunk(spine: blueprint.spine, depth: depth(blueprint))
                    .padding(.top, 14)

                if blueprint.blocks.count > 1 {
                    FlowRow(spacing: 6, trailing: false) {
                        ForEach(blueprint.blocks) { block in
                            range(block, on: block.id == chosen(blueprint)?.id)
                        }
                    }
                    .padding(.top, 12)
                }

                if let block = chosen(blueprint) {
                    BlockPlate(block: block, tokens: blueprint.tokens).padding(.top, 12)
                }
            } else if trouble.isEmpty {
                Text("asking the daemon…").mono(10.5, t.fg3).padding(.vertical, 10)
            } else {
                Refusal(message: trouble).padding(.top, 10)
            }
        }
    }

    private func chosen(_ blueprint: Blueprint) -> BlockGraph? {
        blueprint.blocks.first { $0.id == opened } ?? blueprint.blocks.first
    }

    private func depth(_ blueprint: Blueprint) -> Int {
        blueprint.blocks.reduce(0) { $0 + $1.layers.count }
    }

    private func range(_ block: BlockGraph, on: Bool) -> some View {
        Text(Anatomy.span(block.layers)).mono(11, on ? t.fg : t.fg2)
            .lineLimit(1)
            .padding(.horizontal, 9)
            .padding(.vertical, 3)
            .background(on ? t.accentSoft : .clear)
            .overlay(Capsule().strokeBorder(on ? t.accent : t.hair, lineWidth: 1))
            .clipShape(Capsule())
            .contentShape(Rectangle())
            .onTapGesture { opened = block.id }
    }
}

// ── the trunk ────────────────────────────────────────────────────────────

/// What runs once, in the order it ran. The stack is one stop with its depth on it: a
/// hundred identical boxes would say the same thing at a hundred times the height.
private struct Trunk: View {
    @Environment(\.tokens) private var t
    let spine: [GraphNode]
    let depth: Int

    var body: some View {
        FlowRow(spacing: 0, trailing: false) {
            ForEach(Array(spine.enumerated()), id: \.offset) { index, node in
                HStack(spacing: 0) {
                    if index > 0 {
                        Text("→").mono(11, t.fg3).padding(.horizontal, 5)
                    }
                    stop(node)
                }
            }
        }
    }

    @ViewBuilder
    private func stop(_ node: GraphNode) -> some View {
        let stack = node.role == "stack"
        HStack(spacing: 5) {
            Text(node.label).mono(11, stack ? t.fg : t.fg2, weight: stack ? .medium : .regular)
            if stack { Text("×\(depth)").mono(10, t.accent, weight: .medium) }
        }
        .lineLimit(1)
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(stack ? t.accentSoft : t.elev)
        .overlay(
            RoundedRectangle(cornerRadius: 6)
                .strokeBorder(stack ? t.accent : t.hair2, lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }
}

// ── one block ────────────────────────────────────────────────────────────

private struct BlockPlate: View {
    @Environment(\.tokens) private var t
    let block: BlockGraph
    let tokens: Int

    var body: some View {
        let plan = Anatomy.plan(block)
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 8) {
                Text(block.kind).mono(11.5, t.fg, weight: .medium).lineLimit(1)
                Spacer(minLength: 0)
                Text("\(block.layers.count) layers").mono(10, t.fg3).lineLimit(1)
            }
            .padding(.bottom, 10)

            if !block.kernels.isEmpty {
                Text(block.kernels.joined(separator: " · ")).mono(9.5, t.accent)
                    .lineLimit(1).truncationMode(.middle)
                    .padding(.bottom, 10)
            }

            Canvas { context, size in
                Anatomy.draw(plan, in: context, width: size.width, tokens: t)
            }
            .frame(height: plan.height)
            .frame(maxWidth: .infinity)
        }
    }
}

// ── the layout, and the drawing ──────────────────────────────────────────

enum Anatomy {
    static let rowGap: CGFloat = 26
    static let boxHeight: CGFloat = 25
    static let tallHeight: CGFloat = 37
    static let joinSize: CGFloat = 17
    static let gutter: CGFloat = 9
    static let sideGap: CGFloat = 10

    struct Slot {
        let node: GraphNode
        let row: Int
        var frame: CGRect = .zero
    }

    struct Plan {
        var slots: [Slot]
        var edges: [GraphEdge]
        var rows: [Int: CGFloat]
        var height: CGFloat
    }

    /// Which layers a block runs, as the ranges it runs them in. A hybrid's second kind of
    /// block is `2–34`, and a stack that interleaves is `3, 7, 11, …` — the shape of the
    /// list is itself the answer to how the hybrid alternates.
    static func span(_ layers: [Int]) -> String {
        var runs: [(Int, Int)] = []
        for layer in layers.sorted() {
            if let last = runs.last, layer == last.1 + 1 {
                runs[runs.count - 1].1 = layer
            } else {
                runs.append((layer, layer))
            }
        }
        let said = runs.prefix(4).map { $0.0 == $0.1 ? "\($0.0)" : "\($0.0)–\($0.1)" }
        return said.joined(separator: ", ") + (runs.count > 4 ? ", …" : "")
    }

    /// A node's row is the longest path to it from the block's input, so everything that
    /// reads the same array lands on the same row. Relaxation is bounded by the node count:
    /// the graph is acyclic, and a bound is what keeps a malformed one from spinning.
    static func plan(_ block: BlockGraph) -> Plan {
        var row: [String: Int] = [:]
        for node in block.nodes { row[node.id] = 0 }
        for _ in 0..<block.nodes.count {
            var moved = false
            for edge in block.edges {
                guard let from = row[edge.source], let to = row[edge.target] else { continue }
                if to < from + 1 {
                    row[edge.target] = from + 1
                    moved = true
                }
            }
            if !moved { break }
        }
        // `out` is the foot whatever reached it: a block whose last sub-layer is shallower
        // than another branch would otherwise draw its exit beside a node it comes after.
        row["out"] = (row.values.max() ?? 0) + ((row["out"] ?? 0) == (row.values.max() ?? 0) ? 0 : 1)

        let slots = block.nodes.map { Slot(node: $0, row: row[$0.id] ?? 0) }
        var heights: [Int: CGFloat] = [:]
        for slot in slots {
            let tall = slot.node.kernels.isEmpty ? boxHeight : tallHeight
            heights[slot.row] = max(heights[slot.row] ?? 0, slot.node.role == "join" ? joinSize : tall)
        }
        var tops: [Int: CGFloat] = [:]
        var cursor: CGFloat = 0
        for index in heights.keys.sorted() {
            tops[index] = cursor
            cursor += (heights[index] ?? boxHeight) + rowGap
        }
        return Plan(
            slots: slots,
            edges: block.edges,
            rows: tops,
            height: max(cursor - rowGap, boxHeight)
        )
    }

    static func draw(_ plan: Plan, in context: GraphicsContext, width: CGFloat, tokens t: Tokens) {
        var placed = place(plan, in: context, width: width, tokens: t)
        let frames = Dictionary(uniqueKeysWithValues: placed.map { ($0.node.id, $0.frame) })

        // Edges under the boxes, so a line that has to pass a row is covered by what it
        // passes rather than drawn over it.
        var lane: CGFloat = 0
        for edge in plan.edges {
            guard let from = frames[edge.source], let to = frames[edge.target] else { continue }
            // A skipped row is a residual, and a residual drawn straight down the column
            // would run behind every box between the two ends and read as nothing at all.
            let long = to.minY - from.maxY > rowGap * 1.5
            wire(
                edge, from: from, to: to, aside: long ? lane : nil,
                width: width, in: context, tokens: t
            )
            if long { lane += 1 }
        }
        // A sum the trace only saw one side of is still a sum. Drawn with one arrow it would
        // read as a pass-through, so the side that was not attributed arrives as the stub the
        // dashes already stand for everywhere else in this drawing.
        var arriving: [String: Int] = [:]
        for edge in plan.edges { arriving[edge.target, default: 0] += 1 }
        for slot in placed where slot.node.role == "join" && (arriving[slot.node.id] ?? 0) < 2 {
            var stub = Path()
            stub.move(to: CGPoint(x: slot.frame.minX - 20, y: slot.frame.midY))
            stub.addLine(to: CGPoint(x: slot.frame.minX - 4, y: slot.frame.midY))
            context.stroke(
                stub,
                with: .color(t.hair2),
                style: StrokeStyle(lineWidth: 1, dash: [3, 3])
            )
        }
        for index in placed.indices {
            box(placed[index], in: context, tokens: t)
        }
        placed.removeAll()
    }

    /// Rows are laid out centred: the column reads as one run, and a row that widens pushes
    /// out from the middle instead of shifting everything under it.
    private static func place(
        _ plan: Plan, in context: GraphicsContext, width: CGFloat, tokens t: Tokens
    ) -> [Slot] {
        var slots = plan.slots
        var byRow: [Int: [Int]] = [:]
        for (index, slot) in slots.enumerated() { byRow[slot.row, default: []].append(index) }

        let room = width - 2 * (gutter + sideGap)
        for (row, indices) in byRow {
            var widths: [CGFloat] = []
            for index in indices {
                let slot = slots[index]
                if slot.node.role == "join" {
                    widths.append(joinSize)
                } else if slot.node.role == "port" {
                    widths.append(measure(slot.node.label, size: 10, in: context) + 10)
                } else {
                    let label = measure(slot.node.label, size: 11, in: context)
                    let under = slot.node.kernels.isEmpty
                        ? 0
                        : measure(slot.node.kernels.joined(separator: " · "), size: 9, in: context)
                    widths.append(min(max(label, under) + 16, room))
                }
            }
            let total = widths.reduce(0, +) + sideGap * CGFloat(max(0, indices.count - 1))
            var x = (width - total) / 2
            let top = plan.rows[row] ?? 0
            for (offset, index) in indices.enumerated() {
                let slot = slots[index]
                let height: CGFloat = slot.node.role == "join"
                    ? joinSize : (slot.node.kernels.isEmpty ? boxHeight : tallHeight)
                slots[index].frame = CGRect(x: x, y: top, width: widths[offset], height: height)
                x += widths[offset] + sideGap
            }
        }
        return slots
    }

    private static func measure(
        _ text: String, size: CGFloat, in context: GraphicsContext
    ) -> CGFloat {
        context.resolve(Text(text).font(.system(size: size, design: .monospaced))).measure(
            in: CGSize(width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)
        ).width
    }

    // ── the parts ────────────────────────────────────────────────────────

    private static func box(_ slot: Slot, in context: GraphicsContext, tokens t: Tokens) {
        let frame = slot.frame
        switch slot.node.role {
        case "join":
            context.stroke(Path(ellipseIn: frame), with: .color(t.fg3), lineWidth: 1)
            context.draw(
                Text("+").font(.system(size: 11, design: .monospaced)).foregroundColor(t.fg2),
                at: CGPoint(x: frame.midX, y: frame.midY)
            )
        case "port":
            context.draw(
                Text(slot.node.label).font(.system(size: 10, design: .monospaced))
                    .foregroundColor(t.fg3),
                at: CGPoint(x: frame.midX, y: frame.midY)
            )
        default:
            let quiet = slot.node.role == "norm"
            let marked = !slot.node.kernels.isEmpty
            let shape = Path(roundedRect: frame, cornerRadius: 6)
            context.fill(shape, with: .color(t.elev))
            context.stroke(
                shape,
                with: .color(marked ? t.accent : (quiet ? t.hair : t.hair2)),
                lineWidth: 1
            )
            let label = Text(slot.node.label).font(.system(size: 11, design: .monospaced))
                .foregroundColor(quiet ? t.fg2 : t.fg)
            if marked {
                context.draw(label, at: CGPoint(x: frame.midX, y: frame.minY + 12))
                context.draw(
                    Text(slot.node.kernels.joined(separator: " · "))
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundColor(t.accent),
                    at: CGPoint(x: frame.midX, y: frame.maxY - 11)
                )
            } else {
                context.draw(label, at: CGPoint(x: frame.midX, y: frame.midY))
            }
        }
    }

    /// Down the column when the two are stacked, and out into the gutter when the edge skips
    /// rows — which is the residual: it leaves where the step branches, runs past everything
    /// the branch does, and comes back at the sum.
    private static func wire(
        _ edge: GraphEdge,
        from: CGRect,
        to: CGRect,
        aside lane: CGFloat?,
        width: CGFloat,
        in context: GraphicsContext,
        tokens t: Tokens
    ) {
        var path = Path()
        var head = CGPoint(x: to.midX, y: to.minY - 4)
        var facing = CGVector(dx: 0, dy: 1)
        if let lane {
            // The margin the rows are laid out inside, not an offset from the source box:
            // the widest row is what a gutter line has to clear, and the source may be the
            // narrowest thing in the block.
            let x = gutter - 1 + lane.truncatingRemainder(dividingBy: 3) * 5
            head = CGPoint(x: to.minX - 4, y: to.midY)
            facing = CGVector(dx: 1, dy: 0)
            path.move(to: CGPoint(x: from.midX, y: from.maxY))
            path.addLine(to: CGPoint(x: from.midX, y: from.maxY + 9))
            path.addLine(to: CGPoint(x: x, y: from.maxY + 9))
            path.addLine(to: CGPoint(x: x, y: head.y))
            path.addLine(to: head)
        } else if abs(from.midX - to.midX) < 1 {
            path.move(to: CGPoint(x: from.midX, y: from.maxY))
            path.addLine(to: head)
        } else {
            let mid = (from.maxY + to.minY) / 2
            path.move(to: CGPoint(x: from.midX, y: from.maxY))
            path.addLine(to: CGPoint(x: from.midX, y: mid))
            path.addLine(to: CGPoint(x: to.midX, y: mid))
            path.addLine(to: head)
        }
        context.stroke(
            path,
            with: .color(edge.observed ? t.fg3 : t.hair2),
            style: StrokeStyle(
                lineWidth: 1,
                lineJoin: .round,
                dash: edge.observed ? [] : [3, 3]
            )
        )
        // Turned to the direction it arrives from, so a residual coming in from the side
        // does not point down at the box it is entering.
        let along = CGPoint(x: facing.dx, y: facing.dy)
        let across = CGPoint(x: -facing.dy, y: facing.dx)
        var arrow = Path()
        arrow.move(to: head.shifted(by: along, -4).shifted(by: across, -3.5))
        arrow.addLine(to: head.shifted(by: along, 1))
        arrow.addLine(to: head.shifted(by: along, -4).shifted(by: across, 3.5))
        context.stroke(
            arrow,
            with: .color(edge.observed ? t.fg3 : t.hair2),
            style: StrokeStyle(lineWidth: 1, lineCap: .round, lineJoin: .round)
        )
    }
}

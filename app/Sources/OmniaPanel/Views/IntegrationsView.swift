// Integrations: four checkpoints, named by size, and the line each terminal agent starts
// with to reach them.
//
// The tiers are set here and nowhere else, because the four clients disagree about how many
// slots they have and agree about nothing else — what a client does with a tier is drawn
// under its own row, so the mapping is on screen before the line is copied.

import AppKit
import SwiftUI

struct IntegrationsSection: View {
    @Environment(\.tokens) private var t
    @Bindable var app: AppModel
    @Binding var picking: Tier?

    var body: some View {
        SectionHead(title: "Integrations")
        SettingsGroup {
            ForEach(Array(Tier.allCases.enumerated()), id: \.element) { index, tier in
                let model = app.integrations.model(tier)
                SettingRow(label: tier.label, first: index == 0) {
                    Text(model.map(Fmt.displayName) ?? "not set")
                        .mono(11.5, model == nil ? t.fg3 : t.fg2)
                        .lineLimit(1).truncationMode(.middle)
                    ActionText(label: "Change") { picking = tier }
                }
            }
        }
        SettingNote(text: "A tier with nothing in it is left out of the lines below, and the client keeps whatever it had there.")

        SettingsGroup {
            ForEach(Array(Recipes.all.enumerated()), id: \.element.id) { index, recipe in
                let line = recipe.line(wiring)
                SettingRow(label: recipe.name, first: index == 0) {
                    PushButton(label: "Copy", action: line.map { text in { copy(text) } })
                }
                SettingNote(text: line == nil ? "Nothing set for \(unset(recipe))." : recipe.mapping)
            }
        }
    }

    private var wiring: Wiring {
        Wiring(
            base: Client.base,
            key: app.store.config["api_key"]?.value.text,
            models: app.integrations.models
        )
    }

    /// Which of this client's tiers is empty — the row says what to fill rather than only
    /// that it cannot be copied.
    private func unset(_ recipe: Recipe) -> String {
        recipe.tiers
            .filter { app.integrations.model($0) == nil }
            .map(\.label)
            .map { $0.lowercased() }
            .joined(separator: ", ")
    }

    private func copy(_ line: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(line, forType: .string)
    }
}

/// Which checkpoint a tier names. The chat's own model sheet, minus the image split — what
/// a coding agent needs of a model is not whether it sees.
struct TierSheet: View {
    @Environment(\.tokens) private var t
    @Bindable var app: AppModel
    let tier: Tier
    let close: () -> Void

    var body: some View {
        let listed = app.store.catalog.filter(\.supported)
        let chosen = app.integrations.model(tier)

        Sheet(title: tier.label, close: close) {
            if chosen != nil {
                HStack(spacing: 9) {
                    Text("Leave unset").sans(12.5, t.fg2)
                    Spacer(minLength: 0)
                }
                .contentShape(Rectangle())
                .onTapGesture { pick(nil) }
                .padding(.vertical, 8)
                .padding(.horizontal, 2)
            }
            SectionHead(title: "On disk · \(listed.count)")
            ForEach(sorted(listed)) { entry in row(entry, chosen: entry.id == chosen) }
            if listed.isEmpty {
                EmptyNote(message: "Nothing on disk this engine can load.")
            }
        }
    }

    private func sorted(_ entries: [CatalogEntry]) -> [CatalogEntry] {
        entries.sorted { left, right in
            if left.bytesOnDisk != right.bytesOnDisk { return left.bytesOnDisk < right.bytesOnDisk }
            return left.id < right.id
        }
    }

    private func row(_ entry: CatalogEntry, chosen: Bool) -> some View {
        HStack(spacing: 9) {
            VStack(alignment: .leading, spacing: 2) {
                Text(Fmt.displayName(entry.id)).sans(12.5, t.fg, weight: .medium)
                    .lineLimit(1).truncationMode(.middle)
                Text(under(entry)).mono(10, t.fg3).lineLimit(1)
            }
            Spacer(minLength: 0)
            if chosen { PillView(label: tier.label, kind: "resident") }
        }
        .contentShape(Rectangle())
        .onTapGesture { pick(entry.id) }
        .padding(.vertical, 8)
        .padding(.horizontal, 2)
        .overlay(alignment: .top) { Rectangle().fill(t.hair).frame(height: 1) }
    }

    /// Size first, because that is what a tier is: the list is ordered by it, and the reader
    /// is choosing the smallest thing that will do.
    private func under(_ entry: CatalogEntry) -> String {
        var said = ["\(Fmt.gb(Double(entry.bytesOnDisk))) GB"]
        if let label = entry.quantization { said.append(label) }
        if let decode = app.store.decode(entry.id) { said.append(decode) }
        return said.joined(separator: " · ")
    }

    private func pick(_ model: String?) {
        app.integrations.set(tier, model)
        close()
    }
}

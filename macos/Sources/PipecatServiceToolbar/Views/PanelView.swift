import AppKit
import PipecatToolbarCore
import SwiftUI

/// The menu bar panel: running services, the model catalog, or one service's page.
///
/// Every page has the same size. A `MenuBarExtra` window grows with its content
/// but doesn't reliably shrink back, so a page that changed the panel's height
/// would leave the next, shorter page floating in an oversized window.
struct PanelView: View {
    @Environment(AppModel.self) private var app

    static let width: CGFloat = 360
    static let height: CGFloat = 540

    var body: some View {
        ZStack {
            switch app.page {
            case .running:
                RunningView()
                    .transition(.asymmetric(insertion: .move(edge: .leading), removal: .move(edge: .leading)))
            case .catalog:
                CatalogView()
                    .transition(.move(edge: .trailing))
            case .detail(let service, _):
                ServiceDetailView(model: app.model(service))
                    .transition(.move(edge: .trailing))
            }
        }
        .frame(width: Self.width, height: Self.height)
        .clipped()
        .animation(.snappy(duration: 0.28), value: app.page)
    }
}

// MARK: - Running services

struct RunningView: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        VStack(spacing: 0) {
            header
            if let banner = app.setupError {
                SetupBanner(message: banner)
            } else if app.staleLayout {
                StaleLayoutBanner()
            }
            Divider().opacity(0.5)
            if app.anyActive {
                ScrollView {
                    VStack(alignment: .leading, spacing: 2) {
                        ForEach(ServiceKind.allCases) { kind in
                            let models = app.active(kind)
                            if !models.isEmpty {
                                SectionHeader(kind: kind)
                                ForEach(models) { RunningRow(model: $0) }
                            }
                        }
                    }
                    .padding(.horizontal, 8)
                    .padding(.top, 4)
                    .padding(.bottom, 8)
                }
                .scrollIndicators(.never)
                .frame(maxHeight: .infinity)
            } else {
                EmptyRunningState()
                    .frame(maxHeight: .infinity)
            }
            Divider().opacity(0.5)
            FooterBar(showsAddModels: app.anyActive)
        }
    }

    private var header: some View {
        HStack(spacing: 10) {
            AppGlyph()
            VStack(alignment: .leading, spacing: 1) {
                Text("Pipecat Service Toolbar").font(.system(size: 13.5, weight: .semibold)).lineLimit(1)
                Text(app.summary)
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                    .contentTransition(.numericText())
            }
            Spacer()
            if app.bootstrapping {
                ProgressView().controlSize(.small)
            } else if app.anyInstalling {
                Button {
                    app.page = .catalog
                } label: {
                    Image(systemName: "arrow.down.circle.dotted")
                        .font(.system(size: 15))
                        .foregroundStyle(.blue)
                        .symbolEffect(.pulse)
                }
                .buttonStyle(.plain)
                .help("A model is downloading. Click to see it in Add Models.")
            }
        }
        .padding(.horizontal, 14)
        .padding(.top, 12)
        .padding(.bottom, 10)
    }
}

struct SectionHeader: View {
    let kind: ServiceKind

    var body: some View {
        Label(kind.title, systemImage: kind == .tts ? "speaker.wave.2.fill" : "mic.fill")
            .font(.system(size: 11, weight: .semibold))
            .foregroundStyle(.secondary)
            .padding(.horizontal, 8)
            .padding(.top, 8)
            .padding(.bottom, 2)
    }
}

struct EmptyRunningState: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        VStack(spacing: 10) {
            PipecatMark()
                .fill(.quaternary)
                .frame(width: 72, height: 42)
            Text("No services running")
                .font(.system(size: 13, weight: .semibold))
            Text("Start a speech model to serve it on a local port for your Pipecat apps.")
                .font(.system(size: 11.5))
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.horizontal, 30)
            Button {
                app.page = .catalog
            } label: {
                Label("Add Models", systemImage: "plus")
            }
            .buttonStyle(PillButtonStyle(tint: .accentColor, prominent: true))
            .padding(.top, 2)
        }
        .frame(maxWidth: .infinity)
        .padding(.bottom, 24)
    }
}

/// A started service: endpoint, quick copy, stop.
struct RunningRow: View {
    @Environment(AppModel.self) private var app
    let model: ServiceModel

    var body: some View {
        HStack(spacing: 10) {
            ProviderTile(info: model.info, size: 30)
            VStack(alignment: .leading, spacing: 2) {
                Text(model.info.displayName).font(.system(size: 13, weight: .medium))
                status
            }
            Spacer(minLength: 6)
            trailing
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .contentShape(Rectangle())
        .hoverHighlight()
        .onTapGesture { app.show(model.service) }
        .contextMenu { ServiceMenuItems(model: model) }
    }

    @ViewBuilder private var status: some View {
        HStack(spacing: 5) {
            switch model.phase {
            case .running:
                StatusDot(color: .green)
                Text(verbatim: "127.0.0.1:\(model.port)")
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(.secondary)
            case .starting:
                StatusDot(color: .orange)
                Text(model.info.needsModelDownload ? "Loading model…" : "Starting…")
                    .foregroundStyle(.secondary)
            case .stopping:
                StatusDot(color: .orange)
                Text("Stopping…").foregroundStyle(.secondary)
            case .failed(let failure):
                Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.red).imageScale(.small)
                Text(failure.title).foregroundStyle(.red)
            default:
                EmptyView()
            }
        }
        .font(.system(size: 11))
        .lineLimit(1)
    }

    @ViewBuilder private var trailing: some View {
        switch model.phase {
        case .running:
            HStack(spacing: 2) {
                CopyButton(text: "ws://\(ServiceCatalog.host):\(model.port)/", help: "Copy URL")
                IconButton(systemImage: "stop.fill", help: "Stop") { model.stop() }
            }
        case .starting, .stopping:
            ProgressView().controlSize(.small).frame(width: 44)
        case .failed:
            Button("Retry") { model.start() }.buttonStyle(PillButtonStyle(tint: .red))
        default:
            EmptyView()
        }
    }
}

// MARK: - Catalog ("Add Models")

struct CatalogView: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        @Bindable var app = app
        VStack(spacing: 0) {
            TopBar(title: "Add Models", backTitle: "Running") { app.page = .running }
            Picker("Type", selection: $app.catalogKind) {
                ForEach(ServiceKind.allCases) { kind in
                    Text(kind.title).tag(kind)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .padding(.horizontal, 14)
            .padding(.bottom, 8)
            Divider().opacity(0.5)
            ScrollView {
                VStack(alignment: .leading, spacing: 2) {
                    let active = app.active(app.catalogKind)
                    if !active.isEmpty { group("Running", active) }
                    group("Installed", app.installed(app.catalogKind))
                    group("Available", app.available(app.catalogKind))
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 6)
            }
            .scrollIndicators(.never)
            .frame(maxHeight: .infinity)
            Divider().opacity(0.5)
            FooterBar(showsAddModels: false)
        }
    }

    @ViewBuilder
    private func group(_ title: String, _ models: [ServiceModel]) -> some View {
        if !models.isEmpty {
            Text(title)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.secondary)
                .padding(.horizontal, 8)
                .padding(.top, 8)
                .padding(.bottom, 2)
            ForEach(models) { CatalogRow(model: $0) }
        }
    }
}

struct CatalogRow: View {
    @Environment(AppModel.self) private var app
    let model: ServiceModel

    var body: some View {
        HStack(spacing: 10) {
            ProviderTile(info: model.info)
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 5) {
                    Text(model.info.displayName).font(.system(size: 13, weight: .medium))
                    if model.info.isDefault {
                        Badge(text: "Default", color: .accentColor)
                    }
                    if model.info.licence.isRestrictive {
                        LicenceBadge(licence: model.info.licence)
                    }
                }
                CatalogSubtitle(model: model)
            }
            Spacer(minLength: 6)
            CatalogAction(model: model)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 7)
        .contentShape(Rectangle())
        .hoverHighlight()
        .onTapGesture { app.show(model.service) }
        .contextMenu { ServiceMenuItems(model: model) }
    }
}

struct CatalogSubtitle: View {
    let model: ServiceModel

    var body: some View {
        HStack(spacing: 4) {
            switch model.phase {
            case .checking:
                Text("Checking…").foregroundStyle(.secondary)
            case .unavailable(let why):
                Text(why).foregroundStyle(.secondary)
            case .notInstalled:
                Text(availableText).foregroundStyle(.secondary)
            case .needsModel:
                Text("Model not downloaded").foregroundStyle(.orange)
            case .installing(let stage):
                Text(installingText(stage)).foregroundStyle(.secondary).monospacedDigit()
            case .installed:
                Text(readyText).foregroundStyle(.secondary)
            case .starting:
                Text("Starting…").foregroundStyle(.secondary)
            case .running:
                StatusDot(color: .green)
                Text(verbatim: "Running on :\(model.port)").foregroundStyle(.secondary)
            case .stopping:
                Text("Stopping…").foregroundStyle(.secondary)
            case .removing:
                Text("Deleting…").foregroundStyle(.secondary)
            case .failed(let failure):
                Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.red).imageScale(.small)
                Text(failure.title).foregroundStyle(.red)
            }
        }
        .font(.system(size: 11))
        .lineLimit(1)
    }

    private var availableText: String {
        if let bytes = model.approxDownloadBytes { return "\(Format.bytes(bytes)) · \(model.info.tagline)" }
        return model.info.tagline
    }

    private var readyText: String {
        if let bytes = model.diskBytes { return "Ready · \(Format.bytes(bytes))" }
        return "Ready"
    }

    private func installingText(_ stage: InstallStage) -> String {
        if stage == .model, let d = model.download, d.totalBytes > 0 {
            return "\(Format.bytes(d.downloadedBytes)) of \(Format.bytes(d.totalBytes))"
        }
        if stage == .model, let fraction = model.assetProgress {
            return "Downloading model · \(Int(fraction * 100))%"
        }
        return stage.title + "…"
    }
}

struct CatalogAction: View {
    let model: ServiceModel

    var body: some View {
        switch model.phase {
        case .checking, .starting, .stopping, .removing:
            ProgressView().controlSize(.small).frame(width: 54)
        case .unavailable:
            EmptyView()
        case .notInstalled:
            Button("Get") { model.install() }.buttonStyle(PillButtonStyle())
        case .needsModel:
            Button("Resume") { model.install() }.buttonStyle(PillButtonStyle(tint: .orange))
        case .installing(let stage):
            ProgressRing(fraction: stage == .model ? model.installFraction : nil) { model.cancelInstall() }
                .frame(width: 54)
        case .installed:
            Button("Start") { model.start() }.buttonStyle(PillButtonStyle(tint: .green, prominent: true))
        case .running:
            Button("Stop") { model.stop() }.buttonStyle(PillButtonStyle(tint: .secondary))
        case .failed(let failure):
            switch failure.kind {
            case .install, .download:
                Button("Retry") { model.install() }.buttonStyle(PillButtonStyle(tint: .red))
            case .crashed, .startup, .portBusy:
                Button("Retry") { model.start() }.buttonStyle(PillButtonStyle(tint: .orange))
            }
        }
    }
}

// MARK: - Shared pieces

/// The right-click menu for a service, wherever it appears.
struct ServiceMenuItems: View {
    @Environment(AppModel.self) private var app
    let model: ServiceModel

    var body: some View {
        let info = model.info
        Button("Copy URL") { Pasteboard.copy("ws://\(ServiceCatalog.host):\(model.port)/") }
        Button("Copy Pipecat Service") { Pasteboard.copy(info.pipecatService(port: model.port)) }
        Button("Copy Environment Variables") { Pasteboard.copy(info.environment(port: model.port)) }
        Divider()
        switch model.phase {
        case .running:
            Button("Stop") { model.stop() }
        case .installed:
            Button("Start") { model.start() }
        case .notInstalled, .needsModel:
            Button("Download") { model.install() }
        case .failed(let failure):
            Button(failure.isRuntime ? "Start" : "Retry Download") {
                failure.isRuntime ? model.start() : model.install()
            }
        default:
            EmptyView()
        }
        Button("Show Details") { app.show(model.service) }
        if model.canDeleteModel {
            Divider()
            Button("Delete Model…", role: .destructive) { app.confirmDelete(model) }
        }
    }
}

struct TopBar: View {
    let title: String
    let backTitle: String
    let back: () -> Void

    var body: some View {
        ZStack {
            Text(title).font(.system(size: 13, weight: .semibold))
            HStack {
                Button(action: back) {
                    HStack(spacing: 3) {
                        Image(systemName: "chevron.left").font(.system(size: 11, weight: .semibold))
                        Text(backTitle).font(.system(size: 12.5))
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.borderless)
                .keyboardShortcut(.escape, modifiers: [])
                Spacer()
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
    }
}

struct FooterBar: View {
    @Environment(AppModel.self) private var app
    var showsAddModels = true

    var body: some View {
        HStack(spacing: 10) {
            if showsAddModels {
                Button {
                    app.page = .catalog
                } label: {
                    Label("Add Models", systemImage: "plus.circle")
                        .font(.system(size: 12))
                }
                .buttonStyle(.borderless)
            }
            Spacer()
            Menu {
                Button("Open Logs Folder") { app.openLogsFolder() }
                Button("Open Models Folder") { app.openModelsFolder() }
                Button("Open App Data Folder") { app.openDataFolder() }
                if app.canManageLoginItem {
                    Divider()
                    Toggle("Open at Login", isOn: Binding(get: { app.opensAtLogin }, set: { app.opensAtLogin = $0 }))
                }
            } label: {
                Image(systemName: "gearshape")
                    .font(.system(size: 13))
                    .foregroundStyle(.secondary)
            }
            .menuStyle(.borderlessButton)
            .menuIndicator(.hidden)
            .fixedSize()
            .help("Settings")

            Button {
                NSApp.terminate(nil)
            } label: {
                Text(app.running.isEmpty ? "Quit" : "Stop All & Quit").font(.system(size: 12))
            }
            .buttonStyle(.borderless)
            .keyboardShortcut("q")
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 9)
    }
}

struct SetupBanner: View {
    @Environment(AppModel.self) private var app
    let message: String

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
            VStack(alignment: .leading, spacing: 4) {
                Text(message).font(.system(size: 11.5)).fixedSize(horizontal: false, vertical: true)
                Button("Try Again") { Task { await app.bootstrap() } }
                    .buttonStyle(.link)
                    .font(.system(size: 11.5))
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.orange.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
        .padding(.horizontal, 12)
        .padding(.bottom, 8)
    }
}

struct StaleLayoutBanner: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "arrow.triangle.2.circlepath").foregroundStyle(.orange)
            Text("Environments from another app version were found.")
                .font(.system(size: 11.5))
            Spacer()
            Button("Reset") { app.resetLayout() }.buttonStyle(PillButtonStyle(tint: .orange))
        }
        .padding(10)
        .background(Color.orange.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
        .padding(.horizontal, 12)
        .padding(.bottom, 8)
    }
}

/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
The top-level view routing between the queue dashboard and job setup.
*/

import SwiftUI
import SwiftData

struct ContentView: View {
    @State private var appDataModel: AppDataModel
    @State private var showErrorAlert = false

    init(modelContainerResult: Result<ModelContainer, Error>) {
        _appDataModel = State(initialValue: AppDataModel(
            modelContainerResult: modelContainerResult
        ))
    }

    var body: some View {
        dashboard
            .environment(appDataModel)
            .sheet(isPresented: $appDataModel.showingJobSetup) {
                JobSetupView(existingJob: appDataModel.editingJob)
                    .environment(appDataModel)
            }
            .sheet(isPresented: $appDataModel.showingScheduleSettings) {
                ScheduleSettingsView()
                    .environment(appDataModel)
            }
            .onAppear {
                if appDataModel.state == .error {
                    showErrorAlert = true
                }
            }
            .onChange(of: appDataModel.state) {
                if appDataModel.state == .error {
                    showErrorAlert = true
                }
            }
            .alert(appDataModel.alertMessage, isPresented: $showErrorAlert) {
                Button("OK") {
                    appDataModel.state = .idle
                }
            }
    }

    /// The queue dashboard, wrapped in a navigation stack on iPhone and iPad so
    /// the title and the queue's edit controls have somewhere to live.
    @ViewBuilder
    private var dashboard: some View {
        #if os(macOS)
        QueueDashboardView()
            .navigationTitle("Reality Pulse")
        #else
        NavigationStack {
            QueueDashboardView()
                .navigationTitle("Reality Pulse")
                .navigationBarTitleDisplayMode(.inline)
        }
        #endif
    }
}

/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
The top-level view routing between the queue dashboard and job setup.
*/

import SwiftUI

struct ContentView: View {
    @State private var appDataModel = AppDataModel()
    @State private var showErrorAlert = false

    var body: some View {
        QueueDashboardView()
            .environment(appDataModel)
            .navigationTitle("Reality Pulse")
            .sheet(isPresented: $appDataModel.showingJobSetup) {
                JobSetupView(existingJob: appDataModel.editingJob)
                    .environment(appDataModel)
            }
            .sheet(isPresented: $appDataModel.showingSfMJobSetup) {
                SfMJobSetupView()
                    .environment(appDataModel)
            }
            .sheet(isPresented: $appDataModel.showingGaussianSplatJobSetup) {
                GaussianSplatJobSetupView()
                    .environment(appDataModel)
            }
            .sheet(isPresented: $appDataModel.showingScheduleSettings) {
                ScheduleSettingsView()
                    .environment(appDataModel)
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
}

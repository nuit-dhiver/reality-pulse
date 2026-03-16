/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Configure delayed start and allowed-hours window for the scheduler.
*/

import SwiftUI
import os

private let logger = Logger(subsystem: ObjectCaptureReconstructionApp.subsystem,
                            category: "ScheduleSettingsView")

struct ScheduleSettingsView: View {
    @Environment(AppDataModel.self) private var appDataModel: AppDataModel
    @Environment(\.dismiss) private var dismiss

    @State private var useDelayedStart = false
    @State private var delayedStartDate = Date()
    @State private var useAllowedWindow = false
    @State private var windowStartHour = 22
    @State private var windowEndHour = 6

    var body: some View {
        VStack(spacing: 0) {
            Text("Schedule Settings")
                .font(.headline)
                .padding(.top)

            Divider()
                .padding(.top, 8)

            Form {
                // Delayed start
                Section {
                    Toggle("Delay start until:", isOn: $useDelayedStart)

                    if useDelayedStart {
                        DatePicker("Start at:", selection: $delayedStartDate, in: Date()...,
                                   displayedComponents: [.date, .hourAndMinute])
                    }
                }

                Divider()

                // Allowed window
                Section {
                    Toggle("Only process during allowed hours:", isOn: $useAllowedWindow)

                    if useAllowedWindow {
                        Picker("From:", selection: $windowStartHour) {
                            ForEach(0..<24) { hour in
                                Text(formattedHour(hour)).tag(hour)
                            }
                        }

                        Picker("Until:", selection: $windowEndHour) {
                            ForEach(0..<24) { hour in
                                Text(formattedHour(hour)).tag(hour)
                            }
                        }

                        Text(windowDescription)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .padding()

            Divider()

            HStack {
                Button("Clear") {
                    appDataModel.scheduler.scheduleConfig = ScheduleConfig()
                    appDataModel.scheduler.persist()
                    dismiss()
                }

                Spacer()

                Button("Cancel") {
                    dismiss()
                }

                Button("Save") {
                    var config = ScheduleConfig()
                    if useDelayedStart {
                        config.delayedStart = delayedStartDate
                    }
                    if useAllowedWindow {
                        config.allowedWindowStart = windowStartHour
                        config.allowedWindowEnd = windowEndHour
                    }
                    appDataModel.scheduler.scheduleConfig = config
                    appDataModel.scheduler.persist()
                    dismiss()
                }
                .buttonStyle(.borderedProminent)
            }
            .padding()
        }
        .frame(minWidth: 380, minHeight: 300)
        .onAppear {
            let config = appDataModel.scheduler.scheduleConfig
            useDelayedStart = config.delayedStart != nil
            delayedStartDate = config.delayedStart ?? Date()
            useAllowedWindow = config.hasAllowedWindow
            windowStartHour = config.allowedWindowStart ?? 22
            windowEndHour = config.allowedWindowEnd ?? 6
        }
    }

    private func formattedHour(_ hour: Int) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "h a"
        var components = DateComponents()
        components.hour = hour
        if let date = Calendar.current.date(from: components) {
            return formatter.string(from: date)
        }
        return "\(hour):00"
    }

    private var windowDescription: String {
        if windowStartHour <= windowEndHour {
            return "Processing allowed from \(formattedHour(windowStartHour)) to \(formattedHour(windowEndHour))"
        } else {
            return "Processing allowed from \(formattedHour(windowStartHour)) overnight to \(formattedHour(windowEndHour))"
        }
    }
}

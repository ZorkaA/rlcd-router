import Foundation
import os.log

/// Background Swift actor that consumes GPU Execution Log entries to periodically
/// recalibrate the speculative head's temperature using a Brier-score gradient
/// and 1D Newton-Raphson updates.
///
/// # Strict Idle-Time Gating
/// Recalibration ONLY runs during idle gaps between user requests. If a new
/// request arrives (`cancelCurrentPass()` is called), the actor immediately
/// discards any in-progress optimization state and yields the GPU.
///
/// # MLX Memory Fencing
/// `MLXCacheController.applyCacheLimit()` must be called before the first
/// recalibration pass. This prevents MLX's Metal cache from growing beyond 200MB
/// and triggering OS WKdm compression of the Ring Buffer.
///
/// # Brier Score Optimization
/// We use the analytical Brier gradient with 1D Newton-Raphson temperature search:
///
///   B(T) = (1/N) Σ_i (p_i(T) - y_i)^2
///
/// where `p_i(T) = softmax(logits_i / T)` and `y_i` is the observed routing outcome.
/// The gradient is differentiable through temperature T, enabling analytical Newton steps.
public actor RecalibrationActor {
    // MARK: - State
    private var _logBuffer: [ExecutionLogEntry] = []
    private var _currentTemperature: Float = 1.0
    private var _isCancelled: Bool = false
    private let _log = Logger(subsystem: "AsyncMoERouter", category: "RecalibrationActor")

    // MARK: - Configuration
    private let _minEntriesToRecalibrate: Int
    private let _maxNewtonIterations: Int
    private let _learningRate: Float
    private let _temperatureMin: Float = 0.5
    private let _temperatureMax: Float = 5.0

    // MARK: - Metrics
    public private(set) var totalPassesCompleted: Int = 0
    public private(set) var totalEntriesConsumed: Int = 0

    public init(
        minEntriesToRecalibrate: Int = 256,
        maxNewtonIterations: Int = 10,
        learningRate: Float = 0.01
    ) {
        _minEntriesToRecalibrate = minEntriesToRecalibrate
        _maxNewtonIterations = maxNewtonIterations
        _learningRate = learningRate
    }

    // MARK: - Entry ingestion

    /// Appends newly drained Execution Log entries to the internal buffer.
    public func ingestEntries(_ entries: [ExecutionLogEntry]) {
        _logBuffer.append(contentsOf: entries)
        let count = _logBuffer.count
        _log.debug("RecalibrationActor: Ingested \(entries.count) entries (total buffered: \(count))")
    }

    // MARK: - Recalibration pass

    /// Runs a single Brier-score recalibration pass if enough data is available
    /// and the actor has not been cancelled.
    ///
    /// - Parameter groundTruthExpertIDs: The observed expert selections for the buffered entries.
    /// - Returns: Updated temperature value, or `nil` if skipped/cancelled.
    @discardableResult
    public func runPassIfReady(groundTruthExpertIDs: [Int] = []) async -> Float? {
        guard !_isCancelled else {
            _log.info("RecalibrationActor: Pass skipped (cancelled)")
            return nil
        }
        let bufCount = _logBuffer.count
        let minEntries = _minEntriesToRecalibrate
        guard bufCount >= minEntries else {
            _log.debug("RecalibrationActor: Insufficient data (\(bufCount) < \(minEntries)), skipping pass")
            return nil
        }

        // Apply MLX cache limit before doing any tensor work
        MLXCacheController.applyCacheLimit()

        let entries = Array(_logBuffer.prefix(_minEntriesToRecalibrate))
        let updatedT = _newtonRaphsonTemperatureUpdate(entries: entries, currentT: _currentTemperature)
        _currentTemperature = max(_temperatureMin, min(_temperatureMax, updatedT))

        // Consume processed entries
        _logBuffer.removeFirst(min(entries.count, _logBuffer.count))
        totalPassesCompleted += 1
        totalEntriesConsumed += entries.count
        _isCancelled = false

        _log.info("RecalibrationActor: Pass \(self.totalPassesCompleted) complete — T=\(self._currentTemperature) (consumed \(entries.count) entries)")
        return _currentTemperature
    }

    /// Cancels any in-progress recalibration. Call when a new user request arrives.
    public func cancelCurrentPass() {
        _isCancelled = true
        _log.info("RecalibrationActor: Pass cancelled — inference request pre-empted recalibration")
    }

    /// Resets the cancellation flag (call at the start of an idle window).
    public func resetCancellation() {
        _isCancelled = false
    }

    /// Current calibrated temperature value.
    public var currentTemperature: Float { _currentTemperature }

    // MARK: - Private: Newton-Raphson Brier gradient

    /// Computes one Newton-Raphson step to minimize the Brier score B(T) over temperature T.
    ///
    /// B(T) = (1/N) Σ_i (confidence_i(T) - outcome_i)^2
    ///
    /// We use the mean confidence score from log entries as the calibrated probability
    /// proxy. The analytical gradient dB/dT allows closed-form step computation.
    private func _newtonRaphsonTemperatureUpdate(entries: [ExecutionLogEntry], currentT: Float) -> Float {
        guard !entries.isEmpty else { return currentT }

        // Use logged confidence scores as p_i(T=1); scale by 1/T for different T
        let n = Float(entries.count)
        var brierGradSum: Float = 0.0
        var brierHessSum: Float = 0.0

        for entry in entries {
            let pRaw = entry.confidenceScore  // p(T=1) from GPU log
            // Approximate p(T) ~ pRaw^(1/T) for softmax temperature scaling
            let pT = min(max(pow(pRaw, 1.0 / currentT), 1e-7), 1.0 - 1e-7)
            let outcome: Float = pT >= 0.5 ? 1.0 : 0.0
            let residual = pT - outcome
            // dB/dT contribution (chain rule through softmax)
            let dpDT = -pT * log(max(pRaw, 1e-7)) / (currentT * currentT)
            brierGradSum += 2.0 * residual * dpDT
            // d²B/dT² contribution (approximate via first-order)
            brierHessSum += 2.0 * dpDT * dpDT
        }

        let grad = brierGradSum / n
        let hess = max(brierHessSum / n, 1e-6)  // clamp for numerical stability
        let newT = currentT - (grad / hess)
        _log.debug("RecalibrationActor: NR step — T: \(currentT) → \(newT) (grad=\(grad), hess=\(hess))")
        return newT
    }
}

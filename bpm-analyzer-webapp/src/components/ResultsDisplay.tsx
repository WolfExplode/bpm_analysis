export default function ResultsDisplay() {
  return (
    <div>
      <h2 className="text-2xl font-semibold text-gray-800 mb-4">Analysis Results</h2>
      <div className="space-y-4">
        <p className="text-gray-600">
          The analysis results, including the Plotly chart and summary metrics, will be shown here once the analysis is complete.
        </p>
        {/* Placeholder for plot */}
        <div className="w-full h-96 bg-gray-200 rounded-lg flex items-center justify-center">
          <p className="text-gray-500">BPM Plot</p>
        </div>
        {/* Placeholder for metrics */}
        <div>
          <h3 className="text-lg font-medium text-gray-700">Summary</h3>
          <div className="mt-2 text-gray-600">
            Metrics will be listed here...
          </div>
        </div>
      </div>
    </div>
  );
} 
import Image from "next/image";
import AnalysisForm from "@/components/AnalysisForm";
import ResultsDisplay from "@/components/ResultsDisplay";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-start p-12 bg-gray-50">
      <div className="w-full max-w-4xl">
        <header className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-800">
            Heart Rate Analysis
          </h1>
          <p className="text-lg text-gray-600 mt-2">
            Upload your audio file to analyze the BPM and other metrics.
          </p>
        </header>

        <section className="bg-white p-8 rounded-lg shadow-md border border-gray-200">
          <AnalysisForm />
        </section>

        <section className="mt-8 bg-white p-8 rounded-lg shadow-md border border-gray-200">
          <ResultsDisplay />
        </section>
      </div>
    </main>
  );
}

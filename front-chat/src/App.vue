<template>
  <div class="min-h-screen bg-slate-50">
    <header class="bg-white border-b border-slate-200 sticky top-0 z-40 shadow-sm">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-3">
            <div class="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
              <TruckIcon class="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 class="text-2xl font-bold text-slate-900">
                LogiBrain
              </h1>
              <p class="text-xs text-slate-500 font-medium">Sistema de Analítica Logística</p>
            </div>
          </div>
          
          <div class="flex items-center gap-2 px-4 py-2 rounded-lg border" :class="apiStatus.connected ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'">
            <div class="w-2 h-2 rounded-full" :class="apiStatus.connected ? 'bg-green-600' : 'bg-red-600'"></div>
            <span class="text-xs font-semibold uppercase tracking-wide" :class="apiStatus.connected ? 'text-green-700' : 'text-red-700'">
              {{ apiStatus.connected ? 'Conectado' : 'Desconectado' }}
            </span>
          </div>
        </div>
      </div>
    </header>

    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div class="bg-white rounded-lg p-5 border border-slate-200">
          <div class="flex items-center gap-3">
            <div class="w-12 h-12 bg-blue-600 rounded-lg flex items-center justify-center">
              <BrainIcon class="w-6 h-6 text-white" />
            </div>
            <div>
              <p class="text-xs text-slate-500 font-semibold uppercase tracking-wider">Modelo ML</p>
              <p class="text-lg font-bold text-slate-900">RandomForest</p>
            </div>
          </div>
        </div>

        <div class="bg-white rounded-lg p-5 border border-slate-200">
          <div class="flex items-center gap-3">
            <div class="w-12 h-12 bg-slate-700 rounded-lg flex items-center justify-center">
              <DatabaseIcon class="w-6 h-6 text-white" />
            </div>
            <div>
              <p class="text-xs text-slate-500 font-semibold uppercase tracking-wider">Data Warehouse</p>
              <p class="text-lg font-bold text-slate-900">10K registros</p>
            </div>
          </div>
        </div>

        <div class="bg-white rounded-lg p-5 border border-slate-200">
          <div class="flex items-center gap-3">
            <div class="w-12 h-12 bg-slate-900 rounded-lg flex items-center justify-center">
              <SparklesIcon class="w-6 h-6 text-white" />
            </div>
            <div>
              <p class="text-xs text-slate-500 font-semibold uppercase tracking-wider">Agente IA</p>
              <p class="text-lg font-bold text-slate-900">Groq Llama 3.3</p>
            </div>
          </div>
        </div>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <PredictorML />

        <ChatIA />
      </div>
    </main>

    <footer class="mt-16 py-6 border-t border-slate-200 bg-white">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <p class="text-center text-xs text-slate-500 font-medium">
          LogiBrain © {{ new Date().getFullYear() }} — Sistema de Predicción de Retrasos y Asistente IA Logístico
        </p>
      </div>
    </footer>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { TruckIcon, BrainIcon, DatabaseIcon, SparklesIcon } from 'lucide-vue-next'
import PredictorML from './components/PredictorML.vue'
import ChatIA from './components/ChatIA.vue'
import api from './services/api'

const apiStatus = ref({
  connected: false,
})

onMounted(async () => {
  try {
    await api.healthCheck()
    apiStatus.value.connected = true
  } catch (error) {
    apiStatus.value.connected = false
    console.error('API health check falló:', error)
  }
})
</script>

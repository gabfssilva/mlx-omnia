import React from 'react'
import ReactDOM from 'react-dom/client'
import { App } from './App'
import { followAccent } from './accent'
import './theme.css'

followAccent()

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)

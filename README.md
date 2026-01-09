# 🛒 DSMarket 🛒
# Trabajo Final del Máster en Data Science & IA

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Pandas](https://img.shields.io/badge/Pandas-1.5+-green.svg)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Grupo 2.12 - DSCESP 0925

<a id="tabla-de-contenidos"></a>
## 📋 Tabla de Contenidos

- [Integrantes](#integrantes)
- [Descripción del Proyecto](#descripción-del-proyecto)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Tareas del proyecto](#tareas-del-proyecto)
- [Fuentes de datos](#fuentes-de-datos)
- [Instalación y Configuración](#instalación-y-configuracion)
- [Flujo de Trabajo con Git](#flujo-trabajo-git)
- [Criterios de Evaluación](#criterios-evaluación)
- [Tecnologías Utilizadas](#tecnologias-utilizadas)
- [Licencia](#licencia)

<a id="integrantes"></a>
## 👥 Integrantes
- **Gabriela Alberico** - [GitHub](https://github.com/albericog) | Rama: `feature/gabriela_dev`
- **Jorge Silva** - [GitHub](https://github.com/jsilvazuniga) | Rama: `feature/jorge_dev`
- **Robert Tunzi** - [GitHub](https://github.com/rtunzi) | Rama: `feature/robert_dev`
- **Matias Lannes** - [GitHub](https://github.com/matilannes) | Rama: `feature/matias_dev`
- **Alexis Labrador** - [GitHub](https://github.com/alexisnlh) | Rama: `feature/alexis_dev`

---

<a id="descripción-del-proyecto"></a>
## 🎯  Descripción del Proyecto

DSMarket (anteriormente TradiStores) es una cadena de supermercados en Estados Unidos que busca revolucionar sus procesos mediante transformación digital. Como Nicole, científica de datos sénior del departamento financiero, trabajaremos en la optimización de predicciones de ventas y diversos procesos empresariales mediante técnicas de Data Science e Inteligencia Artificial.

### Contexto del Negocio

DSMarket opera en 3 ciudades principales:
- Nueva York
- Boston
- Filadelfia

El objetivo principal es mejorar las predicciones de ventas que actualmente se realizan con métodos rudimentarios, y optimizar procesos críticos como estimaciones de stock, optimización de precios, entregas, predicciones de stockout.

**[⬆ back to top](#tabla-de-contenidos)**

---

<a id="estructura-del-proyecto"></a>
## 📁 Estructura del Proyecto

```
tfm-capstone-project/
│
├── data/
│   ├── raw/                          # ⚠️ SOLO LECTURA - NO MODIFICAR
│   │   ├── item_sales.csv
│   │   ├── item_prices.csv
│   │   └── daily_calendar_with_events.csv
│   │
│   └── processed/                # Datos procesados y limpios
│
├── notebooks_individuales/                 # Notebooks finales de cada integrante
│   ├── gabriela_alberico.ipynb
│   ├── jorge_silva.ipynb
│   ├── roberto_tunzi.ipynb
│   ├── matias_lannes.ipynb
│   └── alexis_labrador.ipynb
│
├── src/                         # Código fuente del proyecto
│   ├── data/                   # Scripts de carga y procesamiento
│   ├── features/               # Scripts de feature engineering
│   ├── models/                 # Scripts de entrenamiento de modelos
│   └── visualization/          # Scripts de visualización
│
├── models/                     # Modelos entrenados guardados
│
├── reports/                    # Reportes generados
│   └── figures/                # Gráficos y visualizaciones
│
├── Grupo_2_12_dscesp_0925_tfm_ds_market.ipynb              # ⭐ Notebook final para entregar
├── requirements.txt            # Dependencias del proyecto
├── LICENSE                     # Licencia del proyecto
└── README.md                   # Este archivo
```

### ⚠️ Reglas Importantes

- Los archivos CSV grandes están excluidos del control de versiones (.gitignore).
- Asegurarse de documentar cualquier transformación realizada a los datos.
- Los datos raw son **INMUTABLE** - solo lectura.
- Cada integrante trabaja SOLO en su propio notebook individual.
- Documentar apropiadamente todo el código y análisis.
- El `Grupo_2_12_dscesp_0925_tfm_ds_market.ipynb` se construye al final mediante consenso del equipo.

**[⬆ back to top](#tabla-de-contenidos)**

---

<a id="tareas-del-proyecto"></a>
## 📝 Tareas del Proyecto

### Tarea 1: Análisis Exploratorio de Datos (EDA)
Análisis en profundidad del estado actual de DSMarket:
- Análisis de ventas por ciudad (NY, Boston, Filadelfia)
- Identificación de productos populares y variaciones entre ciudades
- Análisis de precios y diferencias entre tiendas
- Creación de dashboard con Business Intelligence

### Tarea 2: Clustering
Identificación de grupos de productos similares:
- Clustering de productos con comportamiento similar
- Clustering de tiendas con características similares
- Definición del número óptimo de clusters (5, 10, 20)
- Evaluación del rendimiento de campañas por cluster

### Tarea 3: Predicción de Ventas
Desarrollo de modelo predictivo de ventas:
- Predicciones a nivel tienda-producto
- Horizonte de predicción: 28 días (4 semanas)
- Agregación de predicciones por departamento/tienda/ciudad
- Evaluación del error predictivo

### Tarea 4: Caso de Uso - Abastecimiento de Tiendas
Aplicación de modelos de ventas para optimización de stock:
- Sistema de reposición semanal de stock
- Minimización de stock remanente
- Propuesta de solución y productivización
- Especificación de API para ejecución

---

<a id="fuentes-de-datos"></a>
## 🗄️ Fuentes de Datos

El proyecto utiliza 3 archivos principales:

1. **daily_calendar_with_events.csv (table: calendar)**: Calendario con eventos
   - Fechas en formato y-m-d
   - Días de la semana
   - Número del día de la semana (Saturday: 1, Friday: 7)
   - Identificador día
   - Eventos especiales

2. **item_prices.csv (table: prices)**: Precios de productos
   - ID de producto
   - Categoría
   - Código de tienda
   - Período de fecha para el precio (formato año-semana)
   - Precio semanal promedio

3. **item_sales.csv (table: sales)**: Ventas de productos
   - ID de serie de ventas (item + store_code)
   - ID de producto
   - Categoría
   - ID departamento (diferente identificador para diferentes tiendas)
   - Nombre de tienda
   - ID de tienda
   - Región
   - Número de unidades vendidas por día (d_1, d_2, ...)

**[⬆ back to top](#tabla-de-contenidos)**

---

<a id="instalación-y-configuracion"></a>
## 🔧 Instalación y Configuración

### Requisitos Previos

- Python 3.8 o superior
- Git instalado y configurado
- Cuenta de GitHub
- Acceso al repositorio (haber aceptado la invitación de colaborador)

### Setup Paso a Paso (TODOS los Integrantes)

#### 1. Clonar el Repositorio
```bash
git clone https://github.com/alexisnlh/tfm-capstone-project.git

# Entrar al directorio
cd tfm-capstone-project
```

#### 2. Verificar Estructura y Ramas
```bash
# Ver en qué rama estás (debería ser main)
git branch

# Ver todas las ramas (remotas y locales)
git branch -a

# Actualizar referencias remotas
git fetch --all
```

#### 3. Crear Entorno Virtual (Recomendado)
```bash
# Crear entorno virtual
python3 -m venv venv

# Activar entorno virtual
# En macOS/Linux:
source venv/bin/activate

# En Windows:
venv\Scripts\activate
```

### 3. Instalar dependencias
```bash
# Instalar todas las librerías necesarias
pip install -r requirements.txt
```

**[⬆ back to top](#tabla-de-contenidos)**

---

<a id="flujo-trabajo-git"></a>
## 🔄 Flujo de Trabajo con Git

### Estrategia de Branching
```
main (protegida - requiere 2 aprobaciones)
├── develop (protegida - solo lectura, se desprotege temporalmente para PR)
├── feature/gabriela_dev
├── feature/jorge_dev
├── feature/robert_dev
├── feature/matias_dev
└── feature/alexis_dev
```

### Comandos Básicos
```bash
# Cambiar a tu rama de desarrollo
git checkout feature/[TU_NOMBRE]_dev

# Hacer cambios y commit
git add .
git commit -m "Descripción de cambios"

# Subir cambios a GitHub
git push origin feature/[TU_NOMBRE]_dev

# Actualizar tu rama con develop
git checkout develop
git pull origin develop
git checkout feature/[TU_NOMBRE]_dev
git merge develop
```

---
<a id="criterios-evaluación"></a>
## 🔍 Criterios de Evaluación

- Desarrollo técnico de las tareas
- Creatividad en las propuestas
- Orientación a negocio
- Comunicación de resultados
- Trabajo en equipo y colaboración

---

<a id="tecnologias-utilizadas"></a>
## 🛠️ Tecnologías Utilizadas

- **Python 3.8+**
- **Pandas** - Manipulación de datos
- **NumPy** - Cálculos numéricos
- **Scikit-learn** - Machine Learning
- **Matplotlib/Seaborn/Plotly** - Visualización
- **Statsmodels/Prophet** - Series temporales

---

<a id="licencia"></a>
## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

## 📧 Contacto

**Alexis Labrador** - [@alexisnlh](https://www.linkedin.com/in/alexisnlh/)

**Link del Proyecto:** [https://github.com/alexisnlh/tfm-capstone-project.git](https://github.com/alexisnlh/tfm-capstone-project.git)

---

**[⬆ back to top](#tabla-de-contenidos)**

<div align="center">

**Nuclio Digital School - Máster en Data Science & IA**

*Proyecto Final - DSMarket - Máster en Data Science & AI - Nuclio Digital School - 2026*

</div>
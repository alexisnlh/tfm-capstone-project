"""
    preparar_datasets_powerbi.py
    ============================
    Script maestro para generar los datasets adicionales para el dashboard
    Power BI del proyecto DSMarket.

    El notebook alexis_labrador.ipynb ya genera los ficheros base:
        - sales_long.parquet       (con store, store_code, city, yearweek, sell_price integrados)
        - store_daily_sales.csv    (store, city, date, total_sales, unique_items)
        - city_daily_sales.csv     (city, date, total_sales, unique_items, weekday, year, month, is_weekend, event)
        - product_daily_sales.csv  (item, category, department, date, total_sales, num_stores)
        - calendar_clean.csv       (date, weekday, weekday_int, event, year, month, day, quarter, week_of_year, is_weekend, is_event)
        - prices_clean.csv         (item, store_code, yearweek, sell_price, yearweek_imputed)
        - replenishment_orders.csv (seccion 12 del notebook)

    Este script genera SOLO lo que falta para cada vista del dashboard:
        - store_daily_sales_revenue.csv  -> Vista Por Tienda (añade revenue al existente)
        - global_metrics.csv             -> Vista Vision General
        - price_analytics.csv            -> Vista Analisis de Precios
        - discount_analysis.csv          -> Vista Analisis de Precios
        - seasonal_index.csv             -> Vista Calendario
        - model_predictions.csv          -> Vista Modelo ML (requiere xgb_tuned.pkl)
        - feature_importance.csv         -> Vista Modelo ML (requiere xgb_tuned.pkl)
        - metrics_by_store.csv           -> Vista Modelo ML (requiere xgb_tuned.pkl)

    REQUISITOS:
        pip install pandas numpy scikit-learn

    USO (desde la raiz del proyecto):
        python src/visualization/preparar_datasets_powerbi.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error


# CONFIGURACION DE RUTAS
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROC = PROJECT_ROOT / 'data' / 'processed'
DATA_VIS = PROJECT_ROOT / 'data' / 'visualization'
MODELS_PATH = PROJECT_ROOT / 'models'
FEAT_PATH = PROJECT_ROOT / 'features'

print("=" * 60)
print("DSMarket - Datasets adicionales para Power BI")
print("=" * 60)
print(f"Leyendo desde: {DATA_PROC}")
print(f"Guardando en: {DATA_VIS}")
print()


# CARGAR DATOS BASE
def load_sales():
    """
        Carga sales_long.parquet generado por el notebook.
        Ya incluye: store, store_code, city, yearweek, sell_price,
            is_weekend, is_event, event, year, month, day, quarter, week_of_year
    """
    path = DATA_PROC / 'sales_long.parquet'

    if not path.exists():
        raise FileNotFoundError(
            "No se encontro sales_long.parquet en data/processed/\n"
            "Ejecuta primero la seccion 7 del notebook (Export Power BI)."
        )

    sales = pd.read_parquet(path)
    sales['date'] = pd.to_datetime(sales['date'])

    print(
        f"   sales_long cargado: {len(sales):,} filas | "
        f"{sales['store'].nunique()} tiendas | "
        f"{sales['item'].nunique():,} productos"
    )

    return sales


def load_prices():
    """
        Carga prices_clean.csv generado por el notebook.
        Columnas: item, store_code, yearweek, sell_price, yearweek_imputed
    """
    path = DATA_PROC / 'prices_clean.csv'

    if not path.exists():
        raise FileNotFoundError(
            "No se encontro prices_clean.csv en data/processed/\n"
            "Ejecuta primero la seccion 7 del notebook (Export Power BI)."
        )

    prices = pd.read_csv(path)
    print(f"   prices_clean cargado: {len(prices):,} filas")

    return prices


# Añade revenue al store_daily_sales.csv del notebook
def create_store_daily_revenue(sales):
    print("\n[1/6] Creando store_daily_sales_revenue.csv...")

    # sales_long ya tiene sell_price integrado -> revenue directo
    sales['revenue'] = sales['sales'] * sales['sell_price'].fillna(0)

    store_daily = sales.groupby(['store', 'city', 'date']).agg(
        total_sales=('sales', 'sum'),
        total_revenue=('revenue', 'sum'),
        unique_items=('item', 'nunique'),
    ).reset_index()

    store_daily['avg_ticket'] = np.where(
        store_daily['total_sales'] > 0,
        store_daily['total_revenue'] / store_daily['total_sales'],
        0
    )

    out = DATA_VIS / 'store_daily_sales_revenue.csv'
    store_daily.to_csv(out, index=False)

    print(f"   Guardado: {out.name} | {len(store_daily):,} filas")
    print(f"   Columnas: {store_daily.columns.tolist()}")


# Vista Vision General: metricas diarias globales
def create_global_metrics(sales):
    print("\n[2/6] Creando global_metrics.csv...")

    sales['revenue'] = sales['sales'] * sales['sell_price'].fillna(0)

    global_daily = sales.groupby('date').agg(
        total_sales=('sales', 'sum'),
        total_revenue=('revenue', 'sum'),
        active_stores=('store', 'nunique'),
        unique_items=('item', 'nunique'),
        is_event=('is_event', 'max'),
    ).reset_index()

    global_daily['avg_ticket'] = np.where(
        global_daily['total_sales'] > 0,
        global_daily['total_revenue'] / global_daily['total_sales'],
        0
    )

    # % de registros con sales = 0 ese dia
    pct_zero = sales.groupby('date').apply(
        lambda x: (x['sales'] == 0).mean()
    ).reset_index(name='pct_zero_sales')

    global_daily = global_daily.merge(pct_zero, on='date', how='left')

    out = DATA_VIS / 'global_metrics.csv'
    global_daily.to_csv(out, index=False)
    print(f"   Guardado: {out.name} | {len(global_daily):,} dias")
    print(f"   Columnas: {global_daily.columns.tolist()}")


# Vista Analisis de Precios
def create_price_analytics(sales, prices):
    print("\n[3/6] Creando price_analytics.csv y discount_analysis.csv...")

    # Metricas de precio por producto (desde prices_clean directamente)
    price_stats = prices.groupby('item').agg(
        avg_price=('sell_price', 'mean'),
        price_std=('sell_price', 'std'),
        price_min=('sell_price', 'min'),
        price_max=('sell_price', 'max'),
    ).reset_index()

    price_stats['price_std'] = price_stats['price_std'].fillna(0)
    price_stats['price_cv'] = np.where(
        price_stats['avg_price'] > 0,
        price_stats['price_std'] / price_stats['avg_price'] * 100,
        0
    )

    # Añadir categoria desde sales_long (ya tiene sell_price integrado)
    cat_map = sales[['item', 'category']].drop_duplicates('item')
    price_stats = price_stats.merge(cat_map, on='item', how='left')

    # Elasticidad precio-demanda simplificada desde sales_long
    sp = sales.sort_values(['item', 'store', 'date']).copy()
    sp['price_pct_chg'] = sp.groupby(['item', 'store'])['sell_price'].pct_change()
    sp['sales_pct_chg'] = sp.groupby(['item', 'store'])['sales'].pct_change()
    valid = sp[
        sp['price_pct_chg'].notna() &
        sp['sales_pct_chg'].notna() &
        np.isfinite(sp['price_pct_chg']) &
        np.isfinite(sp['sales_pct_chg']) &
        (sp['price_pct_chg'] != 0) &
        (sp['price_pct_chg'].abs() < 0.5)
    ]
    elasticity = valid.groupby('item').apply(
        lambda x: (x['sales_pct_chg'] / x['price_pct_chg']).mean()
    ).reset_index(name='price_elasticity')
    # Reemplazar cualquier inf/-inf residual por NaN (compatible con Power BI)
    elasticity['price_elasticity'] = elasticity['price_elasticity'].replace(
        [np.inf, -np.inf], np.nan
    )
    price_stats = price_stats.merge(elasticity, on='item', how='left')

    # Margen estimado por categoria (simulado - no disponible en el proyecto)
    def get_margin(cat):
        cat = str(cat).upper()
        if 'FOOD' in cat or 'SUPERMARKET' in cat: return 0.20
        elif 'GARDEN' in cat or 'HOME' in cat or 'HOBBY' in cat: return 0.35
        else: return 0.30

    price_stats['estimated_margin'] = price_stats['category'].apply(get_margin)
    np.random.seed(42)
    price_stats['estimated_margin'] *= (1 + np.random.uniform(-0.05, 0.05, len(price_stats)))

    # Tier de precio
    price_stats['price_tier'] = pd.cut(
        price_stats['avg_price'],
        bins=[0, 2, 4, 6, 100],
        labels=['Low', 'Medium', 'High', 'Premium']
    ).astype(str)

    out1 = DATA_VIS / 'price_analytics.csv'
    price_stats.to_csv(out1, index=False)
    print(f"   Guardado: {out1.name} | {len(price_stats):,} productos")

    # Descuento vs precio base (precio mas frecuente = precio base)
    price_base = prices.groupby('item')['sell_price'].agg(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else x.mean()
    ).reset_index(name='base_price')

    prices_base = prices.merge(price_base, on='item')
    prices_base['discount_pct'] = (
        (prices_base['base_price'] - prices_base['sell_price']) /
        prices_base['base_price'] * 100
    ).clip(lower=0)

    discount = prices_base.groupby('item').agg(
        avg_discount_pct=('discount_pct', 'mean'),
        base_price=('base_price', 'first')
    ).reset_index()

    out2 = DATA_VIS / 'discount_analysis.csv'
    discount.to_csv(out2, index=False)
    print(f"   Guardado: {out2.name} | {len(discount):,} productos")


# Vista Calendario: indice de estacionalidad mensual
def create_seasonal_index(sales):
    print("\n[4/6] Creando seasonal_index.csv...")

    monthly = sales.groupby(['year', 'month'])['sales'].sum().reset_index(name='total_sales')
    monthly_avg = monthly.groupby('month')['total_sales'].mean().reset_index(name='avg_all_years')
    seasonal = monthly.merge(monthly_avg, on='month')
    seasonal['seasonal_index'] = seasonal['total_sales'] / seasonal['avg_all_years']

    out = DATA_VIS / 'seasonal_index.csv'
    seasonal.to_csv(out, index=False)
    print(f"   Guardado: {out.name} | {len(seasonal):,} filas (año x mes)")
    print(f"   Columnas: {seasonal.columns.tolist()}")


# model_predictions.csv + feature_importance.csv + metrics_by_store.csv
def create_ml_datasets():
    print("\n[5/6] Creando datasets de ML...")

    model_path = MODELS_PATH / 'best_model.pkl'
    preds_path = DATA_VIS / 'model_predictions.csv'
    fi_path = DATA_VIS / 'feature_importance.csv'
    ms_path = DATA_VIS / 'metrics_by_store.csv'

    # Si ya existen los tres, no regenerar
    if preds_path.exists() and fi_path.exists() and ms_path.exists():
        print("  Los tres ficheros ML ya existen. Omitiendo regeneracion.")
        return

    if not model_path.exists():
        print("   ADVERTENCIA: No se encontro best_model.pkl en models/")
        print("   Guarda el modelo desde el notebook y vuelve a ejecutar:")
        print()
        print("       import pickle")
        print("       with open(f'{MODELS_PATH}/best_model.pkl', 'wb') as f:")
        print("           pickle.dump(xgb_tuned, f)")
        print()
        print("   O genera los CSV directamente en el notebook (ver GUIA_VISTA_MODELO_ML.md sec 2)")
        return

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("   Modelo cargado desde best_model.pkl")

        val_path = DATA_PROC / 'validation_set.parquet'
        if not val_path.exists():
            print("   ADVERTENCIA: No se encontro validation_set.parquet")
            print("   Se genera en la seccion 9 del notebook (Export ML).")
            return

        df_val = pd.read_parquet(val_path)

        # Cargar la lista exacta de features usada en el notebook
        feat_json = FEAT_PATH / 'feature_list.json'
        if not feat_json.exists():
            print(f"   ADVERTENCIA: No se encontro features_list.json en {FEAT_PATH}")
            print("   Se genera en la seccion 9 del notebook (Export ML).")
            return

        with open(feat_json) as f:
            feature_cols = json.load(f)['features']
        print(f"   Features cargadas desde features_list.json: {len(feature_cols)}")

        X_val = df_val[feature_cols]
        y_true = df_val['sales'].values
        y_pred = np.clip(model.predict(X_val), 0, None)

        # Identificar columna de tienda
        store_col = next(
            (c for c in ['store', 'store_code', 'store_id'] if c in df_val.columns),
            None
        )

        # model_predictions.csv
        results = df_val[['date', 'item', store_col, 'category', 'sales']].copy()
        results = results.rename(columns={'sales': 'actual_sales', store_col: 'store_code'})
        results['predicted_sales'] = y_pred
        results['error']           = results['actual_sales'] - results['predicted_sales']
        results['abs_error']       = results['error'].abs()
        results['pct_error']       = np.where(
            results['actual_sales'] > 0,
            results['abs_error'] / results['actual_sales'] * 100,
            np.nan
        )
        results['squared_error'] = results['error'] ** 2

        def quality(pct):
            if pd.isna(pct):  return 'Sin_Venta'
            elif pct < 20:    return 'Excelente'
            elif pct < 40:    return 'Bueno'
            elif pct < 70:    return 'Normal'
            else:             return 'Revisar'

        results['prediction_quality'] = results['pct_error'].apply(quality)
        results.to_csv(preds_path, index=False)
        print(f"  Guardado: {preds_path.name} | {len(results):,} filas")

        # feature_importance.csv
        fi = pd.DataFrame({
            'feature':    feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        fi['rank']           = range(1, len(fi) + 1)
        fi['importance_pct'] = fi['importance'] / fi['importance'].max() * 100

        def feat_type(name):
            if any(x in name for x in ['lag', 'rolling']):              return 'Historico'
            elif any(x in name for x in ['price', 'sell']):             return 'Precio'
            elif any(x in name for x in ['day', 'week', 'month',
                                          'year', 'weekend', 'event',
                                          'quarter']):                   return 'Temporal'
            else:                                                        return 'Producto/Tienda'

        fi['feature_type'] = fi['feature'].apply(feat_type)
        fi.to_csv(fi_path, index=False)
        print(f"  Guardado: {fi_path.name} | {len(fi)} features")

        # metrics_by_store.csv
        store_metrics = []
        for store in results['store_code'].unique():
            df_s = results[results['store_code'] == store]
            rmse = np.sqrt(mean_squared_error(df_s['actual_sales'], df_s['predicted_sales']))
            mask = df_s['actual_sales'] > 0
            mape = df_s.loc[mask, 'pct_error'].mean() if mask.sum() > 0 else np.nan
            store_metrics.append({'store_code': store, 'rmse': round(rmse, 4), 'mape': round(mape, 2) if not np.isnan(mape) else None})
        pd.DataFrame(store_metrics).to_csv(ms_path, index=False)
        print(f"  Guardado: {ms_path.name} | {len(store_metrics)} tiendas")

    except Exception as e:
        print(f"  ERROR: {e}")
        print("  Genera los CSV manualmente desde el notebook (GUIA_VISTA_MODELO_ML.md sec 2)")


# Verificar ficheros del notebook + resumen
def print_summary():
    print("\n" + "=" * 60)
    print("RESUMEN DE TODOS LOS FICHEROS PARA POWER BI")
    print("=" * 60)

    files = [
        # Generados por el notebook (no tocar)
        ('sales_long.parquet', 'NOTEBOOK s7', 'Todas las vistas (base)'),
        ('calendar_clean.csv', 'NOTEBOOK s7', 'Vista Calendario'),
        ('prices_clean.csv', 'NOTEBOOK s7', 'Vista Analisis Precios'),
        ('store_daily_sales.csv', 'NOTEBOOK s7', 'Vista Por Tienda (sin revenue)'),
        ('city_daily_sales.csv', 'NOTEBOOK s7', 'Vista Vision General'),
        ('product_daily_sales.csv', 'NOTEBOOK s7', 'Vista Por Producto'),
        ('replenishment_orders.csv', 'NOTEBOOK s12', 'Vista Modelo ML - Abastecimiento'),

        # Generados por este script
        ('store_daily_sales_revenue.csv', 'ESTE SCRIPT', 'Vista Por Tienda (con revenue)'),
        ('global_metrics.csv', 'ESTE SCRIPT', 'Vista Vision General'),
        ('price_analytics.csv', 'ESTE SCRIPT', 'Vista Analisis Precios'),
        ('discount_analysis.csv', 'ESTE SCRIPT', 'Vista Analisis Precios'),
        ('seasonal_index.csv', 'ESTE SCRIPT', 'Vista Calendario'),
        ('model_predictions.csv', 'ESTE SCRIPT', 'Vista Modelo ML'),
        ('feature_importance.csv', 'ESTE SCRIPT', 'Vista Modelo ML'),
        ('metrics_by_store.csv', 'ESTE SCRIPT', 'Vista Modelo ML'),
    ]

    print(f"{'FICHERO':<40} {'ORIGEN':<15} {'ESTADO':<6} {'VISTA'}")
    print("-" * 100)

    notebook_files = {f[0] for f in files if f[1].startswith('NOTEBOOK')}
    for fname, origin, vista in files:
        path = DATA_PROC / fname if fname in notebook_files else DATA_VIS / fname
        status = "OK" if path.exists() else "FALTA"
        size = f"({path.stat().st_size // 1024} KB)" if path.exists() else ""
        print(f"   {fname:<40} {origin:<15} {status:<6} {size:<10} {vista}")

    print()
    print("NOTA: model_predictions/feature_importance/metrics_by_store requieren xgb_tuned.pkl en models/ para generarse.")
    print()
    print("SIGUIENTE PASO: Abrir Power BI e importar segun RESUMEN_EQUIPO.md")


if __name__ == '__main__':
    try:
        print("\n[0/6] Cargando datos base del notebook...")
        sales = load_sales()
        prices = load_prices()

        create_store_daily_revenue(sales)
        create_global_metrics(sales)
        create_price_analytics(sales, prices)
        create_seasonal_index(sales)
        create_ml_datasets()
        print_summary()

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nAsegurate de:")
        print("  1. Haber ejecutado el notebook alexis_labrador.ipynb completo")
        print("  2. Ejecutar este script desde el directorio raiz del proyecto:")
        print("     python src/visualization/preparar_datasets_powerbi.py")

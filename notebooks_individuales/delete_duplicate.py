from pathlib import Path
import pandas as pd


BASE_PATH = Path.cwd().parent

def get_duplicate_prices():
    # Cargar prices_clean
    prices = pd.read_csv(f'{BASE_PATH}/data/processed/prices_clean.csv')

    # Crear la key igual que en Power BI
    prices['key'] = prices['item'].astype(str) + '_' + prices['store_code'].astype(str) + '_' + prices['yearweek'].astype(str)

    # Buscar duplicados
    duplicados = prices[prices.duplicated(subset=['key'], keep=False)].sort_values('key')

    print(f"Total registros: {len(prices)}")
    print(f"Keys únicas: {prices['key'].nunique()}")
    print(f"Duplicados: {len(duplicados)}")

    # Ver ejemplos de duplicados
    print("\n5 primeros duplicados:")
    print(duplicados[['item', 'store_code', 'yearweek', 'sell_price', 'key']].head(10))

    # Contar cuántos duplicados por key
    dup_counts = prices['key'].value_counts()
    dup_counts = dup_counts[dup_counts > 1]
    print(f"\nKeys con más de 1 registro: {len(dup_counts)}")
    print(f"Máximo de duplicados para una key: {dup_counts.max()}")


def create_dedup():
    # Cargar prices
    prices = pd.read_csv(f'{BASE_PATH}/data/processed/prices_clean.csv')

    print(f"Registros originales: {len(prices):,}")

    # Eliminar duplicados (mantener el primero)
    # Como todos tienen el mismo precio, da igual cuál mantener
    prices_limpio = prices.drop_duplicates(
        subset=['item', 'store_code', 'yearweek'], 
        keep='first'
    )

    print(f"Registros después de limpiar: {len(prices_limpio):,}")
    print(f"Eliminados: {len(prices) - len(prices_limpio):,} duplicados")

    # Guardar archivo limpio
    prices_limpio.to_csv(f'{BASE_PATH}/data/processed/prices_clean_dedup.csv', index=False)
    print("Guardado: prices_clean_dedup.csv")

    # Verificar que no hay duplicados
    check = prices_limpio.duplicated(subset=['item', 'store_code', 'yearweek']).sum()
    print(f"Verificación: {check} duplicados restantes (debería ser 0)")


def info_prices():
    prices = pd.read_csv(f'{BASE_PATH}/data/processed/prices_clean_dedup.csv')

    print("=== Tipos de datos ===")
    print(prices.dtypes)

    print("\n=== sell_price estadísticas básicas ===")
    print(prices['sell_price'].describe())

    print("\n=== Valores extremos de sell_price ===")
    print(prices.nlargest(10, 'sell_price')[['item', 'store_code', 'yearweek', 'sell_price']])

    print("\n=== Valores más pequeños de sell_price ===")
    print(prices.nsmallest(10, 'sell_price')[['item', 'store_code', 'yearweek', 'sell_price']])

    print("\n=== Valores con sell_price > 1000 (sospechosos) ===")
    sospechosos = prices[prices['sell_price'] > 1000]
    print(f"Total registros sospechosos: {len(sospechosos)}")
    print(sospechosos.head(10)[['item', 'store_code', 'yearweek', 'sell_price']])

    # Verificar si hay un patrón en los valores corruptos
    print("\n=== Distribución de sell_price ===")
    print(prices['sell_price'].value_counts().sort_index().head(20))

    # Verificar si los precios originales (antes de dedup) tenían el mismo problema
    prices_original = pd.read_csv(f'{BASE_PATH}/data/processed/prices_clean.csv')
    print("\n=== sell_price original estadísticas ===")
    print(prices_original['sell_price'].describe())

    sospechosos_original = prices_original[prices_original['sell_price'] > 1000]
    print(f"\nRegistros sospechosos en original: {len(sospechosos_original)}")


def get_revenue_sales():
    df_sales_long = pd.read_parquet(f'{BASE_PATH}/data/processed/sales_long.parquet')
    print(len(df_sales_long))
    print(df_sales_long.shape)
    df_sales_long['revenue'] = df_sales_long['sales'] * df_sales_long['sell_price']


if __name__ == "__main__":
    get_duplicate_prices()
    create_dedup()
    info_prices()
    get_revenue_sales()
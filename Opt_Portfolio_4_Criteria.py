import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

#Importamos los datos
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "JPM", "JNJ", "NVDA", "XOM",
    "V", "PG", "UNH", "HD", "BAC", "DIS", "MA", "PYPL", "ADBE", "NFLX",
    "KO", "CSCO", "PFE", "NKE", "CRM", "MRK", "ABT", "PEP", "INTC", "T",
    "WMT", "CVX", "ORCL", "COST", "QCOM", "ACN", "AVGO", "TXN", "LLY", "MCD",
    "MDT", "NEE", "HON", "PM", "AMGN", "DHR", "IBM", "LOW", "SBUX", "BMY"
]

START = datetime(2000,1,1)
    
database = yf.download(TICKERS, start = START)
    
#Solo queremos el precio ajustado de las acciones
database = database['Close']
    
#Eliminamos lo factores faltantes y calculamos los rendimientos diarios
data = database.dropna().pct_change(1).dropna()

#Separamos los datos, el primer 70% paraentrenamiento y el 30% restante para backtestear.
split = int(0.7*len(data))
train_set = data.iloc[:split, :]
test_set = data.iloc[split:,:]


def MV_criterion(weights, data):

    #El input de esta función son los rendimientos de los activos y los pesos.
    #El output es la optimización del criterio de la cartera.

    #Parámetros
    Lambda = 3
    W = 1
    Wbar = 1 + 0.25/100

    #Calcular la Rentabilidad de la Cartera
    portfolio_return = np.multiply(data, np.transpose(weights))
    portfolio_return = portfolio_return.sum(axis=1)

    #Calculamos la media y la volatilidad de la cartera
    mean = np.mean(portfolio_return, axis = 0)
    std = np.std(portfolio_return, axis = 0)

    #Calcuamos el criterio de Media-Varianza
    criterion = Wbar**(1-Lambda)/(1+Lambda)+\
                    Wbar**(-Lambda)*W*mean-\
                    Lambda/2*Wbar**(-1-Lambda)*\
                    W**2*std**2

    criterion = -criterion

    return criterion

def SK_criterion(weights, data):

    #El input de esta función son los rendimientos de los activos y los pesos.
    #El output es la optimización del criterio de la cartera.

    #Parámetros
    Lambda = 3
    W = 1
    Wbar = 1 + 0.25/100

    #Calcular la Rentabilidad de la Cartera
    portfolio_return = np.multiply(data, np.transpose(weights))
    portfolio_return = portfolio_return.sum(axis=1)

    #Calculamos la media y la volatilidad de la cartera, el sesgo y la kurtosis de la cartera
    mean = np.mean(portfolio_return, axis = 0)
    std = np.std(portfolio_return, axis = 0)

    skewness = skew(portfolio_return, 0)

    kurt = kurtosis(portfolio_return, 0) 

    #Calculamos el Criterio
    criterion = Wbar**(1-Lambda)/(1+Lambda) + Wbar**(-Lambda)\
     *W*mean - Lambda/2 * Wbar**(-1-Lambda)*W**2*std**2\
     +Lambda*(Lambda+1)/(6)*Wbar**(-2-Lambda)*W**3*skewness\
     -Lambda*(Lambda+1)*(Lambda+2)/(24)*Wbar**(-3-Lambda)*\
     W**4*kurt

    criterion = -criterion

    return criterion

def SR_criterion(weights,data):

    #El input de esta función son los rendimientos de los activos y los pesos.
    #El output es el opuesto del Ratio de Sharpe para minimizarlo

    #Activo libre de riesgo
    rf = 0

    #Calcular la Rentabilidad de la Cartera
    portfolio_return = np.multiply(data, np.transpose(weights))
    portfolio_return = portfolio_return.sum(axis=1)

    #Calculamos la media y la volatilidad de la cartera
    mean = np.mean(portfolio_return, axis = 0)
    std = np.std(portfolio_return, axis = 0)

    #Calculamos el ratio de Sharpe
    sharpe = (mean - rf)/std

    #Opuesto del Ratio de Sharpe
    sharpe = -sharpe

    return sharpe

def SOR_criterion(weights, data):

    #El input de esta función son los rendimientos de los activos y los pesos.
    #El output es el opuesto del Ratio de Sortino para minimizarlo.

    #Activo libre de riesgo
    rf = 0

    #Calcular la Rentabilidad de la Cartera
    portfolio_return = np.multiply(data, np.transpose(weights))
    portfolio_return = portfolio_return.sum(axis=1)

    #Calculamos la media y la volatilidad de la cartera
    mean = np.mean(portfolio_return, axis = 0)
    std = np.std(portfolio_return[portfolio_return<0], axis = 0) #Calculamos solo la volatilidad cuando, portfolio_return<0 

    #Calcular el Ratio de Sortino
    sortino = (mean - rf)/std

    #Opuesto del Ratio de Sortino
    sortino = -sortino

    return sortino


def optimize(criterion):

    #Para saber el número de activos
    n = data.shape[1]
    
    #Inicialización de los pesos
    x0 = np.ones(n)
    
    #Restricciones del problema de optimización
    cons = ({'type':'eq','fun':lambda x:sum(x)-1})
    
    #Establecer los límites
    Bounds = [(0,1)for i in range(0,n)]
    
    #Resolvemos el problema de optimización
    res = minimize(criterion, x0, method = 'SLSQP', 
                      args = (train_set), bounds = Bounds, 
                      constraints = cons, options = {'disp':True})

    #Pesos Optimos
    X = res.x

    return X

X_MV = optimize(MV_criterion)
X_SK = optimize(SK_criterion)
X_SR= optimize(SR_criterion)
X_SOR= optimize(SOR_criterion)

#Rentabilidad Acumulada de la cartera Media-Varianza
portfolio_return_MV = np.multiply(test_set, np.transpose(X_MV))
portfolio_return_MV = portfolio_return_MV.sum(axis=1)

#Rentabilidad Acumulada de la cartera SK
portfolio_return_SK = np.multiply(test_set, np.transpose(X_SK))
portfolio_return_SK = portfolio_return_SK.sum(axis=1)

#Rentabilidad Acumulada de la cartera Ratio de Sharpe
portfolio_return_SR = np.multiply(test_set, np.transpose(X_SR))
portfolio_return_SR = portfolio_return_SR.sum(axis=1)

#Rentabilidad Acumulada de la cartera Ratio de Sortino
portfolio_return_SOR = np.multiply(test_set, np.transpose(X_SOR))
portfolio_return_SOR = portfolio_return_SOR.sum(axis=1)

#REPRESENTACIÓN GRÁFICA
plt.figure(figsize=(12,8))
plt.plot(np.cumsum(portfolio_return_MV)*100, color='#035593', linewidth=3, label='Media-Varianza')
plt.plot(np.cumsum(portfolio_return_SK)*100, color='green', linewidth=3, label='SK')
plt.plot(np.cumsum(portfolio_return_SR)*100, color='red', linewidth=3, label='Sharpe Ratio')
plt.plot(np.cumsum(portfolio_return_SOR)*100, color='black', linewidth=3, label='Sortino Ratio')
plt.ylabel("Cummulative Return %", size=15, fontweight='bold')
plt.xticks(size=15, fontweight='bold')
plt.yticks(size=15, fontweight='bold')
plt.title("Cummulative Return of the Mean-Variance Portfolio", size=15)
plt.axhline(0, color='r',linewidth=3)
plt.legend()
plt.savefig("Cumulative_Return.png", dpi=300, bbox_inches='tight')
plt.show()

#Guardamos los resultados en DataFrames

# Pesos de las carteras

#Le damos formato de porcentaje
pd.set_option('display.float_format', lambda x: f'{x:,.2f}%')

df_weights = pd.DataFrame({
    'Ticker': TICKERS,
    'MV_Weights': X_MV*100,
    'SK_Weights': X_SK*100,
    'SR_Weights': X_SR*100,
    'SOR_Weights': X_SOR*100
})

# Rentabilidad esperada (diaria)
returns_mean = {
    'MV': ((1 + np.mean(portfolio_return_MV))**252 - 1)*100,
    'SK': ((1 + np.mean(portfolio_return_SK))**252 - 1)*100,
    'SR': ((1 + np.mean(portfolio_return_SR))**252 - 1)*100,
    'SOR': ((1 + np.mean(portfolio_return_SOR))**252 - 1)*100
}

df_returns = pd.DataFrame.from_dict(returns_mean, orient='index', columns=['Expected Return - Annualized'])

# Mostrar los dataframes
display(df_weights)
display(df_returns)

#Guardamos los dataframes en un excel

with pd.ExcelWriter("Results.xlsx", engine='xlsxwriter') as writer:
    df_weights.to_excel(writer, sheet_name="Weights", index=False)
    df_returns.to_excel(writer, sheet_name="Returns", index=True)


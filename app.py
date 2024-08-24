import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Configuración de la página de Streamlit
st.title("Simulación de Evolución de Cartera con DCA en el S&P 500")

st.write('''En esta simulación hemos utilizado el retorno medio anual compuesto del S&P500 de los últimos 35 años 
         de acuerdo con TradingView (11% anual) así como la volatilidad observada para el mismo periodo (15,26% de desviación típica.)
         Existe la opción de descontar el impacto de la inflación esperada. Los bancos centrales establecen el objetivo en el 2% anual,
         aunque nosotros por defecto aplicamos un 2,42% que ha sido el valor medio para los últimos 20 años en los EEUU, y una desviación típica de 1,24%. Puedes modificar este dato a tu gusto. 
         Si se aplica la inflación, esta se descuenta del retorno anual y a su vez se revaloriza la aportación mensual anualmente.
            ''')

# Controles deslizantes para ajustar los parámetros
contribution = st.sidebar.slider('Contribución mensual (€)', min_value=0, max_value=1000, step=50)
avg_inflation = st.sidebar.slider('Inflación anual (%)', min_value=0.0, max_value=10.0, value=2.42, step=0.1)
years = st.sidebar.slider('Número de años', min_value=1, max_value=100, step=1)
simulations = st.sidebar.number_input('Número de simulaciones', 1, 10000, 1, 1)


def simulate_portfolio(contr,avg_pi,period,num_sims):
    # Checkbox para activar o desactivar la inflación
    apply_inflation = st.sidebar.checkbox('Aplicar ajuste inflacionario', value=True)

    # Inicialización de las listas para guardar resultados
    final_cap = []
    all_portfolios = []
    distr_inflation = []
    contributions = []

    # Simulaciones
    for sim in range(num_sims): 
        portfolio_value = [1000]  # Valor inicial de la cartera
        inflation = np.random.normal(avg_pi/100, 0.0124, period) # Inflación anual
        series = np.random.normal(0.008735, 0.044052, 12 * period) # Retorno anual mensualizado           
        y = 0
        adjusted = []
        rate = 0
        distr_inflation.extend(inflation)
        total_contributed = [1000]
        # Monthly iteration
        for r in range(0,len(series)):
            # Updating contribution
            if r == 0:
                contribution = contr
            if r % 12 == 0 and apply_inflation:
                rate = inflation[y]
                contribution *= (1 + rate)
                y += 1
            # Updates real return series
            adjusted.append(series[r] - ((1 + rate)**(1/12) - 1))
            total_contributed.append(contribution)   

            if apply_inflation:
                new_value = contribution + portfolio_value[r] * (1 + adjusted[r])
            else:
                new_value = contribution + portfolio_value[r] * (1 + series[r])

            portfolio_value.append(new_value)
            last_contribution = contribution
        contributions.append(total_contributed)
        final_cap.append(new_value)
        all_portfolios.append(portfolio_value)

    # Calcular la curva de Gauss para la distribución de la inflación
    mu, std = avg_pi / 100, 0.0124
    xmin, xmax = np.min(distr_inflation), np.max(distr_inflation)
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    
    return final_cap, all_portfolios, last_contribution, distr_inflation, x, p

final_cap, all_portfolios, last_contribution, distr_inflation, x, p = simulate_portfolio(contribution,avg_inflation,years,simulations)

# Gráfico de la evolución de la cartera
plt.figure(figsize=(10,6))
for portfolio in all_portfolios:
    plt.plot(portfolio)

# Mostrar estadísticas
mediana_capital = np.median(final_cap)
media_capital = np.mean(final_cap)
st.write(f"**Mediana del capital final después de {years} año/s:** {mediana_capital:,.2f}€")
st.write(f"**Media del capital final después de {years} año/s:** {media_capital:,.2f}€")
st.write(f"**Promedio de correlación entre las simulaciones:** {np.mean(np.corrcoef(all_portfolios)):.4f}")

# Mostrar gráfico de evolución de la cartera
plt.title(f'Evolución de la Cartera en {years} Año/s (Contribución mensual final de {"{:,.2f}€".format(last_contribution)})')
plt.xlabel('Meses')
plt.ylabel('Valor de la Cartera (€ en valor presente)')
st.pyplot(plt)

# Histograma de la distribución de la inflación
plt.figure(figsize=(10, 6))
count, bins, ignored = plt.hist(distr_inflation, bins=15, density=True, alpha=0.6, color='g')

# Dibujar la curva de Gauss sobre el histograma
plt.plot(x, p, 'k', linewidth=2)
title = f'Distribución de la inflación en simulaciones\n$\mu={avg_inflation:.2f}$%, $\sigma={0.0124*100:.2f}$%'
plt.title(title)
plt.xlabel('Tasa de inflación')
plt.ylabel('Frecuencia')

# Mostrar el gráfico del histograma en Streamlit
st.pyplot(plt)

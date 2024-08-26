import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Configuración de la página de Streamlit
st.title("Simulación de Evolución de Cartera con DCA (Dollar Cost Averaging) en el S&P 500 :chart:")

with st.expander('🛠️ Guía de Uso'):
    # Guía de uso
    st.markdown("""
    Esta aplicación te permite simular la evolución de una cartera de inversión en el S&P 500 utilizando la estrategia de DCA (Dollar Cost Averaging). A continuación te explico cómo utilizarla:

    1. **Contribución mensual (€):** Ajusta el monto que deseas aportar mensualmente a tu cartera.
    2. **Inflación anual (%):** Configura la tasa de inflación esperada. Puedes optar por aplicar este ajuste inflacionario o no.
    3. **Número de años:** Selecciona el periodo durante el cual deseas realizar la simulación.
    4. **Número de simulaciones:** Indica cuántas simulaciones deseas ejecutar para ver la variabilidad de los resultados.
    5. **Aplicar ajuste inflacionario:** Puedes optar por aplicar un ajuste por inflación a tus contribuciones anuales.
    6. **Lanzar simulación:** Haz clic en este botón para ejecutar la simulación con los parámetros seleccionados.

    Los resultados mostrarán tanto el valor nominal como el valor ajustado por inflación de tu cartera a lo largo del tiempo. También podrás ver un histograma con la distribución de las tasas de inflación observadas durante las simulaciones.

    """)

st.write('''En esta simulación he utilizado el retorno medio anual compuesto del S&P500 de los últimos 35 años 
         de acuerdo con TradingView (11% anual) así como la volatilidad observada para el mismo periodo (15,26% de desviación típica.)
         Existe la opción de descontar el impacto de la inflación esperada. Los bancos centrales establecen el objetivo en el 2% anual,
         aunque nosotros por defecto aplicamos un 2,42% que ha sido el valor medio para los últimos 20 años en los EEUU, y una desviación típica de 1,24%. Puedes modificar este dato a tu gusto. 
         Si se aplica la inflación, esta se descuenta del retorno anual y a su vez se revaloriza la aportación mensual anualmente.
            ''')

# Controles deslizantes para ajustar los parámetros
contribution = st.sidebar.slider('Contribución mensual (€)', min_value=0, max_value=10000, step=50)
avg_inflation = st.sidebar.slider('Inflación anual (%)', min_value=0.0, max_value=20.0, value=2.42, step=0.1)
years = st.sidebar.slider('Número de años', min_value=1, max_value=100, step=1)
simulations = st.sidebar.number_input('Número de simulaciones', 1, 10000, 1, 1)

def simulate_portfolio(contr, avg_pi, period, num_sims):
    # Checkbox para activar o desactivar la inflación
    apply_inflation = st.sidebar.checkbox('Aplicar ajuste inflacionario', value=True)
    
    # Inicialización de las listas para guardar resultados
    final_cap = []
    all_portfolios = []
    distr_inflation = []
    contributions = []
    inflation_adjusted_line = []
    all_npv_portfolios = []

    # Simulaciones
    for sim in range(num_sims):
        portfolio_value = [1000]  # Valor inicial de la cartera
        inflation = np.random.normal(avg_pi / 100, 0.0124, period)  # Inflación anual
        series = np.random.normal(0.008735, 0.044052, 12 * period)  # Retorno anual mensualizado
        y = 0
        adjusted = []
        rate = 0
        distr_inflation.extend(inflation)
        total_contributed = [1000]
        inflation_adjusted_value = [1000]  # Track the inflation-adjusted value without investment

        # Monthly iteration
        for r in range(0, len(series)):
            # Updating contribution
            if r == 0:
                contribution = contr
            if r % 12 == 0 and apply_inflation:
                rate = inflation[y]
                contribution *= (1 + rate)
                y += 1
            # Updates real return series
            adjusted.append(series[r] - ((1 + rate) ** (1 / 12) - 1))
            total_contributed.append(contribution)

            # Calculate the new portfolio value with investment
            if apply_inflation:
                new_value = contribution + portfolio_value[r] * (1 + adjusted[r])
            else:
                new_value = contribution + portfolio_value[r] * (1 + series[r])

            portfolio_value.append(new_value)
            last_contribution = contribution

            # Calculate the inflation-adjusted value without investment
            new_inflation_adjusted_value = inflation_adjusted_value[-1] + contribution
            inflation_adjusted_value.append(new_inflation_adjusted_value)

        contributions.append(total_contributed)
        final_cap.append(new_value)
        all_portfolios.append(portfolio_value)

        # Calculating NPV where year 0 = 100
        acc_inflation = [1]  # Start with 1 (no inflation impact in year 0)
        for i in range(1, period):
            acc_inflation.append(acc_inflation[-1] * (1 + inflation[i]))

        # Stretch the accumulated inflation to the monthly granularity
        acc_inflation_monthly = np.repeat(acc_inflation, 12)

        # Calculate NPV using the adjusted inflation and portfolio values
        npv_portfolio = [np.log(portfolio_value[i] / acc_inflation_monthly[i // 12]) for i in range(len(portfolio_value))]
        all_npv_portfolios.append(npv_portfolio)

        if apply_inflation:
            inflation_adjusted_line.append(inflation_adjusted_value)

    # Calcular la curva de Gauss para la distribución de la inflación
    mu, std = avg_pi / 100, 0.0124
    xmin, xmax = np.min(distr_inflation), np.max(distr_inflation)
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)

    return final_cap, all_portfolios, last_contribution, distr_inflation, x, p, inflation_adjusted_line, all_npv_portfolios

final_cap, all_portfolios, last_contribution, distr_inflation, x, p, inflation_adjusted_line, all_npv_portfolios = simulate_portfolio(contribution, avg_inflation, years, simulations)

button = st.sidebar.button('Lanzar simulación')

if button:
    # Gráfico de la evolución de la cartera
    plt.figure(figsize=(10, 6))
    for portfolio in all_portfolios:
        plt.plot(portfolio)  # Plot each simulated portfolio

    # Plot the inflation-adjusted contributions line if selected
    if inflation_adjusted_line:
        plt.plot(np.mean(inflation_adjusted_line, axis=0), color='black', linewidth=2, label='Capital inicial + contribuciones indexadas')

    # Mostrar estadísticas
    mediana_capital = np.median(final_cap)
    media_capital = np.mean(final_cap)
    st.write(f"**Mediana del capital final después de {years} año/s:** {mediana_capital:,.2f}€")
    st.write(f"**Media del capital final después de {years} año/s:** {media_capital:,.2f}€")
    st.write(f"**Promedio de correlación entre las simulaciones:** {np.mean(np.corrcoef(all_portfolios)):.4f}")

    # Mostrar gráfico de evolución de la cartera
    plt.title(f'Valor nominal de la Cartera en {years} Año/s (Contribución mensual final de {"{:,.2f}€".format(last_contribution)})')
    plt.xlabel('Meses')
    plt.ylabel('Valor de la Cartera (€)')
    plt.legend()
    st.pyplot(plt)

    st.write('La linea negra representa la evolución de tu capital de no estar invertido. Por lo tanto, si las lineas de simulación se encuentran por encima de la negra, hemos observado un retorno real sobre la inversión positivo. Dado que el retorno nominal del S&P500 tiene la media establecida en el 11%, prueba subir la inflación por encima de este valor para ver como el retorno real se vuelve negativo. (Simulaciones por debajo de la linea negra)')
    st.write('Retorno real = Retorno Nominal - Inflación')
    st.write('Bajo se encuentra la evolución del capital en Logaritmos y la distribución de tasas de inflación observadas. ')
    # Gráfico de la evolución de la cartera
    plt.figure(figsize=(10, 6))
    for portfolio in all_portfolios:
        plt.plot(np.log(portfolio))  # Plot each simulated portfolio

    # Plot the inflation-adjusted contributions line if selected
    if inflation_adjusted_line:
        plt.plot(np.mean(np.log(inflation_adjusted_line), axis=0), color='black', linewidth=2, label='Capital inicial + contribuciones indexadas')

    plt.title(f'Valor de la Cartera en {years} Año/s (Logs)')
    plt.xlabel('Meses')
    plt.ylabel('Valor de la Cartera en Logs')
    st.pyplot(plt)


    # Histograma de la distribución de la inflación
    plt.figure(figsize=(10, 6))
    count, bins, ignored = plt.hist(distr_inflation, bins=15, density=True, alpha=0.6, color='g')

    # Dibujar la curva de Gauss sobre el histograma
    plt.plot(x, p, 'k', linewidth=2)
    title = f'Distribución de la inflación en simulaciones\n$\mu={avg_inflation:.2f}$%, $\sigma={0.0124 * 100:.2f}$%'
    plt.title(title)
    plt.xlabel('Tasa de inflación')
    plt.ylabel('Frecuencia')

    # Mostrar el gráfico del histograma en Streamlit
    st.pyplot(plt)

# Sidebar Contact Information
st.sidebar.markdown("### Contact & Support")
st.sidebar.markdown("""
- 🐦 [Twitter](https://twitter.com/_paugbp)
- 💼 [LinkedIn](https://www.linkedin.com/in/paugb/)
- ☕️ [Buy Me a Coffee](https://buymeacoffee.com/paugb?new=1)
""")

# Social Media and Contact Information
st.markdown("""
**Contact Me & Support**  
Feel free to reach out to me on my social networks or buy me a coffee if you enjoy this app! ☕️

- 🐦 **Twitter**: [@_Paugbp](https://twitter.com/_paugbp)
- 💼 **LinkedIn**: [Paugb](https://linkedin.com/in/paugb)
- ☕️ **[Buy Me a Coffee](https://buymeacoffee.com/paugb?new=1)**  
""")
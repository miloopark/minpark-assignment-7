from flask import Flask, render_template, request, url_for, session
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  

def generate_data(N, mu, beta0, beta1, sigma2, S):
    # Generate data and initial plots
    X = np.random.uniform(low=0.0, high=1.0, size=N)
    error_term = np.random.normal(loc=0, scale=np.sqrt(sigma2), size=N)
    Y = beta0 + beta1 * X + mu + error_term

    # Fit a linear regression model to X and Y
    X_reshaped = X.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X_reshaped, Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    # Generate a scatter plot of (X, Y) with the fitted regression line
    plot1_path = "static/plot1.png"
    plt.figure()
    plt.scatter(X, Y, label='Data Points')
    plt.plot(X, model.predict(X_reshaped), color='red', label='Regression Line')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot with Regression Line')
    plt.legend()
    plt.savefig(plot1_path)
    plt.close()

    # Run S simulations to generate slopes and intercepts
    slopes = []
    intercepts = []
    for _ in range(S):
        X_sim = np.random.uniform(low=0.0, high=1.0, size=N)
        error_term_sim = np.random.normal(loc=0, scale=np.sqrt(sigma2), size=N)
        Y_sim = beta0 + beta1 * X_sim + mu + error_term_sim

        X_sim_reshaped = X_sim.reshape(-1, 1)
        sim_model = LinearRegression()
        sim_model.fit(X_sim_reshaped, Y_sim)
        sim_slope = sim_model.coef_[0]
        sim_intercept = sim_model.intercept_

        slopes.append(sim_slope)
        intercepts.append(sim_intercept)

    # Plot histograms of slopes and intercepts
    plot2_path = "static/plot2.png"
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].hist(slopes, bins=30, color='blue', alpha=0.7)
    axs[0].axvline(slope, color='red', linestyle='dashed', linewidth=2, label='Observed Slope')
    axs[0].set_title('Histogram of Slopes')
    axs[0].set_xlabel('Slope')
    axs[0].set_ylabel('Frequency')
    axs[0].legend()
    axs[1].hist(intercepts, bins=30, color='green', alpha=0.7)
    axs[1].axvline(intercept, color='red', linestyle='dashed', linewidth=2, label='Observed Intercept')
    axs[1].set_title('Histogram of Intercepts')
    axs[1].set_xlabel('Intercept')
    axs[1].set_ylabel('Frequency')
    axs[1].legend()
    plt.tight_layout()
    plt.savefig(plot2_path)
    plt.close()

    # Return data needed for further analysis, including slopes and intercepts
    return (
        X,
        Y,
        slope,
        intercept,
        plot1_path,
        plot2_path,
        slopes,
        intercepts,
    )

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input from the form
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        beta0 = float(request.form["beta0"])
        beta1 = float(request.form["beta1"])
        S = int(request.form["S"])

        if S <= 1:
            return "Number of simulations (S) must be greater than 1 to calculate confidence interval", 400

        # Generate data and initial plots
        (
            X,
            Y,
            slope,
            intercept,
            plot1,
            plot2,
            slopes,
            intercepts,
        ) = generate_data(N, mu, beta0, beta1, sigma2, S)

        # Store data in session
        session["X"] = X.tolist()
        session["Y"] = Y.tolist()
        session["slope"] = float(slope)
        session["intercept"] = float(intercept)
        session["slopes"] = [float(s) for s in slopes]
        session["intercepts"] = [float(i) for i in intercepts]
        session["N"] = N
        session["mu"] = mu
        session["sigma2"] = sigma2
        session["beta0"] = beta0
        session["beta1"] = beta1
        session["S"] = S

        return render_template(
            "index.html",
            plot1=plot1,
            plot2=plot2,
            N=N,
            mu=mu,
            sigma2=sigma2,
            beta0=beta0,
            beta1=beta1,
            S=S,
        )
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    # This route handles data generation (same as index route)
    return index()

@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    # Retrieve data from session
    N = int(session.get("N"))
    S = int(session.get("S"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))

    parameter = request.form.get("parameter")
    test_type = request.form.get("test_type")

    if not parameter or not test_type:
        return "Parameter and test type are required", 400

    if parameter == "slope":
        simulated_stats = np.array(slopes)
        observed_stat = slope
        hypothesized_value = beta1
    else:
        simulated_stats = np.array(intercepts)
        observed_stat = intercept
        hypothesized_value = beta0

    # Calculate p-value based on test type
    if test_type == ">":
        p_value = np.mean(simulated_stats > hypothesized_value)
    elif test_type == "<":
        p_value = np.mean(simulated_stats < hypothesized_value)
    elif test_type == "!=":
        p_value = np.mean(np.abs(simulated_stats - hypothesized_value) >= np.abs(observed_stat - hypothesized_value))
    else:
        p_value = None

    if p_value is not None and p_value <= 0.0001:
        fun_message = "Wow! You've encountered a rare event with p-value ≤ 0.0001!"
    else:
        fun_message = None

    # Plot histogram of simulated statistics
    plot3_path = "static/plot3.png"
    plt.figure()
    plt.hist(simulated_stats, bins=30, color='skyblue', alpha=0.7)
    plt.axvline(observed_stat, color='red', linestyle='dashed', linewidth=2, label='Observed Statistic')
    plt.axvline(hypothesized_value, color='green', linestyle='dotted', linewidth=2, label='Hypothesized Value')
    plt.xlabel(parameter.capitalize())
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Simulated {parameter.capitalize()}s')
    plt.legend()
    plt.savefig(plot3_path)
    plt.close()

    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot3=plot3_path,
        parameter=parameter,
        observed_stat=observed_stat,
        hypothesized_value=hypothesized_value,
        N=N,
        p_value=p_value,
        fun_message=fun_message,
    )

@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    # Retrieve data from session
    N = int(session.get("N"))
    mu = float(session.get("mu"))
    sigma2 = float(session.get("sigma2"))
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))
    S = int(session.get("S"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")

    parameter = request.form.get("parameter")
    confidence_level = float(request.form.get("confidence_level"))

    if not parameter or confidence_level is None:
        return "Parameter and confidence level are required", 400

    if parameter == "slope":
        estimates = np.array(slopes)
        true_param = beta1
    else:
        estimates = np.array(intercepts)
        true_param = beta0

    mean_estimate = np.mean(estimates)
    std_estimate = np.std(estimates, ddof=1)
    std_error = std_estimate / np.sqrt(S)

    if std_error == 0:
        return "Standard error is zero, confidence interval cannot be computed", 400

    alpha = 1 - (confidence_level / 100)
    df = S - 1
    t_critical = stats.t.ppf(1 - alpha / 2, df)

    ci_lower = mean_estimate - t_critical * std_error
    ci_upper = mean_estimate + t_critical * std_error
    includes_true = ci_lower <= true_param <= ci_upper

    # Plot the individual estimates as gray points and confidence interval
    plot4_path = "static/plot4.png"
    plt.figure(figsize=(10, 2))
    plt.scatter(estimates, np.zeros_like(estimates), color='gray', alpha=0.5, label='Estimates')
    plt.scatter(mean_estimate, 0, color='blue', s=100, label='Mean Estimate')
    ci_color = 'green' if includes_true else 'red'
    plt.hlines(0, ci_lower, ci_upper, colors=ci_color, linewidth=5, label=f'{confidence_level}% Confidence Interval')
    plt.axvline(true_param, color='orange', linestyle='dashed', linewidth=2, label='True Parameter')
    plt.xlabel(parameter.capitalize())
    plt.yticks([])
    plt.title(f'{int(confidence_level)}% Confidence Interval for {parameter.capitalize()}')
    plt.legend()
    plt.savefig(plot4_path)
    plt.close()

    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot4=plot4_path,
        parameter=parameter,
        confidence_level=confidence_level,
        mean_estimate=mean_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        includes_true=includes_true,
        true_param=true_param
    )

if __name__ == "__main__":
    app.run(debug=True)

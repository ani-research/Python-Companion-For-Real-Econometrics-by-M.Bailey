import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

class AnStat:

    def __init__(self, ols):
        self.ols = ols

    def give_vif(self, drop_const=True):

        X = self.ols.model.exog
        names = self.ols.model.exog_names

        vif = pd.DataFrame({
            "Variable": names,
            "VIF": [variance_inflation_factor(X, i)
                    for i in range(X.shape[1])]
        })

        if drop_const:
            vif = vif[vif["Variable"] != "Intercept"]

        return vif.reset_index(drop=True)

    def coefficient_plot(self,
                     figsize=(6,4),
                     color="#0060ac",
                     title="Regression coefficients",
                     show_zero=True):

        coef = self.ols.params[1:]
        ci = self.ols.conf_int().iloc[1:]

        y = np.arange(len(coef))

        plt.figure(figsize=figsize)

        plt.errorbar(
            coef,
            y,
            
            xerr=[
                coef - ci[0],
                ci[1] - coef
            ],
            fmt="o",
            capsize=0,
            color=color
        )

        if show_zero:
            plt.axvline(0, color="gray", linestyle="--")

        plt.axvline(0, color="gray", linestyle="--")

        plt.yticks(y, coef.index)

        plt.xlabel("Estimated coefficient")
        plt.title("Regression coefficients")

        plt.show()
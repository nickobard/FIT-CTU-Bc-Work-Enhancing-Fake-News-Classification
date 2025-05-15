"""
Adapted in full from:
Flajžík, J. (2024). Classification of Fake News in the Media/Social media  Ecosystem (Bachelor’s thesis).
Dept. of Applied Mathematics, Czech Technical University in Prague, Faculty of Information Technology.
URL: https://dspace.cvut.cz/bitstream/handle/10467/115641/F8-BP-2024-Flajzik-Jan-thesis.pdf?sequence=-1&isAllowed=y (accessed 12/2/2024)
"""


def export_results_to_latex(self, model: str) -> None:
    '''
    Exports results of experiment to latex table and prints some info about experiments to console.

    Parameters
    ----------
    model: string
            Classification model used for experiments.

    Returns
    -------
    None
    '''

    df = self.results_per_model[self.current_model]
    latex_table = df.to_latex(index=False, escape=False)
    # some minor changes in latex appearance
    latex_table = latex_table.replace('\\toprule', '')
    latex_table = latex_table.replace('\\bottomrule', '')
    latex_table = latex_table.replace('\\midrule', '\\midrule \\midrule')
    latex_table = latex_table.replace('{lllll}', '{l||c|c|c|c}')

    acc_max = df['Accuracy'].astype(float).idxmax()
    f1_max = df['F1-Score'].astype(float).idxmax()

    print('Preprocessing with highest accuracy: ' + str(df.at[acc_max, 'Preprocessing']))
    print('Preprocessing with highest f1-score: ' + str(df.at[f1_max, 'Preprocessing']))
    print("")

    with open(self.dataset_path + 'tables/' + model + '_experiments_results.txt', 'w') as f:
        f.write(latex_table)
        f.close()
    return list(df.loc[acc_max])

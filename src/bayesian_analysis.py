import pickle
import networkx as nx
from matplotlib import pyplot as plt
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork


# Funzione che visualizza il grafo del Bayesian Network
def visualizeBayesianNetwork(bayesianNetwork: BayesianNetwork):
    G = nx.MultiDiGraph(bayesianNetwork.edges())
    pos = nx.spring_layout(G, iterations=100, k=2,
                           threshold=5, pos=nx.spiral_layout(G))
    nx.draw_networkx_nodes(G, pos, node_size=150, node_color="#ff574c")
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=10,
        font_weight="bold",
        clip_on=True,
        horizontalalignment="center",
        verticalalignment="bottom",
    )
    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowsize=7,
        arrowstyle="->",
        edge_color="purple",
        connectionstyle="angle3,angleA=90,angleB=0",
        min_source_margin=1.2,
        min_target_margin=1.5,
        edge_vmin=2,
        edge_vmax=2,
    )

    plt.title("BAYESIAN NETWORK GRAPH")
    plt.show()
    plt.clf()

def visualizeInfo(bayesianNetwork: BayesianNetwork):
    # Ottengo le distribuzioni di probabilità condizionata (CPD)
    cpd_list = bayesianNetwork.get_cpds()
    for cpd in cpd_list:
        print(f"\nCPD per la variabile '{cpd.variable}':")
        print(cpd)
        print("=" * 40)


# Funzione che crea la rete bayesiana
def bayesianNT(dataSet):
    # Passo 1: Creare una rete iniziale utilizzando HillClimbSearch
    hc=HillClimbSearch(dataSet)
    model_structure=hc.estimate(scoring_method='k2score')

    # Passo 3: Creare il modello BayesianNetwork con la struttura aggiornata
    model = BayesianNetwork(model_structure.edges())
    model.fit(dataSet, estimator=MaximumLikelihoodEstimator, n_jobs=-1)

    with open('modello.pkl', 'wb') as output:
        pickle.dump(model, output)
    visualizeBayesianNetwork(model)

    return model

# Funzione che carica la rete bayesiana da file
def load_model():
    with open('modello.pkl', 'rb') as input:
        model = pickle.load(input)
    visualizeBayesianNetwork(model)
    # visualizeInfo(model)
    return model

# Predico il valore di differentialColumn per l'esempio
def bayesian_prediction(bayesianNetwork: BayesianNetwork, example, differentialColumn):
    inference = VariableElimination(bayesianNetwork)
    result = inference.query(variables=[differentialColumn], evidence=example)
    print(result)

# Genera un esempio randomico
def generate_example(bayesianNetwork: BayesianNetwork):
    return bayesianNetwork.simulate(n_samples=1).drop(columns=['Exam_Score'])
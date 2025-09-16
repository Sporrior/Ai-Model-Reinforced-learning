# AdvancedSkittleAgent - Uitleg en Implementatie

Dit document geeft een uitgebreide uitleg van de `AdvancedSkittleAgent` uit `agent.py`. Het behandelt de architectuur, gebruikte technieken, trainingsproces, en best practices.

---

## Inhoud

- [Introductie](#introductie)  
- [Architectuur van het Neural Network](#architectuur-van-het-neural-network)  
- [Belangrijke Hyperparameters](#belangrijke-hyperparameters)  
- [Actie Selectie - Epsilon-Greedy Strategie](#actie-selectie---epsilon-greedy-strategie)  
- [Ervaring Geheugen en Replay](#ervaring-geheugen-en-replay)  
- [Double DQN Training](#double-dqn-training)  
- [Loss Functie](#loss-functie)  
- [Model Opslaan en Laden](#model-opslaan-en-laden)  
- [Mogelijke Verbeteringen en Tips](#mogelijke-verbeteringen-en-tips)  
- [Samenvatting](#samenvatting)  

---

## Introductie

De `AdvancedSkittleAgent` is een Deep Q-Network (DQN) agent ontworpen voor het `SkittleEnvironment`. Het doel is om via reinforcement learning een strategie te leren om de hoogste cumulatieve beloning te behalen.

Deze agent maakt gebruik van een Double DQN aanpak, waardoor hij stabieler en nauwkeuriger leert door het verminderen van overschatting in Q-waardes.

---

## Architectuur van het Neural Network

Het netwerk bestaat uit meerdere lagen:

- **Input Layer:** Ontvangt de huidige omgevingstoestand (`state_size` features).
- **Hidden Layers:**  
  - Dense laag met 128 units + ReLU + L2-regularisatie (0.001)  
  - Batch Normalization + Dropout (0.2) voor betere generalisatie  
  - Dense laag met 64 units + ReLU + L2-regularisatie (0.001)  
  - Batch Normalization + Dropout (0.2)  
  - Dense laag met 32 units + ReLU  
  - Batch Normalization
- **Output Layer:**  
  - Dense laag met `action_size` units, lineaire activatie, die Q-waardes voor elke actie voorspelt

**Optimizer:** Adam met learning rate 0.00025 en gradient clipping (clipnorm=1.0) om stabiel te trainen.

**Loss:** Mean Squared Error (MSE).

---

## Belangrijke Hyperparameters

| Parameter        | Waarde  | Omschrijving                                      |
|------------------|---------|--------------------------------------------------|
| `gamma`          | 0.99    | Discount factor voor toekomstige beloningen      |
| `epsilon`        | 1.0     | Startwaarde voor exploratie (random acties)      |
| `epsilon_min`    | 0.01    | Minimale exploratie (exploiteren)                |
| `epsilon_decay`  | 0.998   | Factor waarmee epsilon per trainingsstap daalt   |
| `learning_rate`  | 0.00025 | Leer snelheid van het netwerk                    |
| `batch_size`     | 64      | Aantal samples per trainingsbatch                |
| `memory_size`    | 10.000  | Grootte van replay buffer (ervaring geheugen)    |
| `update_target_every` | 100 | Aantal trainingsstappen waarna target model wordt geüpdatet |

---

## Actie Selectie - Epsilon-Greedy Strategie

Tijdens training kiest de agent:

- Met kans `epsilon` een random actie (exploratie)  
- Anders de actie met de hoogste Q-waarde (exploitatie)

Epsilon daalt langzaam gedurende training om steeds meer te vertrouwen op geleerde kennis.

---

## Ervaring Geheugen en Replay

De agent slaat ervaringen op als tuples:

```python
(state, action, reward, next_state, done)
```

Deze worden opgeslagen in een deque met maximale lengte `memory_size`.

Tijdens training wordt een random minibatch uit dit geheugen gesampled om het netwerk te trainen — dit voorkomt correlaties tussen opeenvolgende ervaringen en stabiliseert training.

**Prioritized Replay:** Hoewel TD-errors worden berekend en opgeslagen, wordt op dit moment geen prioritized sampling toegepast. Dit kan later worden toegevoegd voor verbeterde leer-efficiëntie.

---

## Double DQN Training

Om overschatting van Q-waardes tegen te gaan, gebruikt de agent Double DQN:

1. Bepaal `next_actions` via het huidige model:
```python
next_actions = np.argmax(self.model.predict(next_states), axis=1)
```

2. Bepaal de Q-waarden van die acties via het target model:
```python
next_q = self.target_model.predict(next_states)
```

Targets worden dan:
```
target = reward + γ × Q_target(next_state, argmaxₐ Q_model(next_state, a)) × (1 - done)
```

Train het model met deze targets.

---

## Loss Functie

Huidig wordt Mean Squared Error (MSE) gebruikt als loss functie.

**Alternatief:** Huber loss kan stabieler zijn en minder gevoelig voor outliers.

---

## Model Opslaan en Laden

**Opslaan:**
- Modelgewichten worden opgeslagen in `.weights.h5` bestanden
- De agent status (epsilon, replay geheugen, TD-errors, trainingsstap) wordt opgeslagen als `.npy` bestand

**Laden:**
- Gewichten en agent status kunnen worden geladen om training te hervatten of om te evalueren

---

## Mogelijke Verbeteringen en Tips

- **Implementatie van Prioritized Experience Replay:** Gebruik TD-errors om vaker ervaringen met grote fout te herhalen, wat de leersnelheid kan verbeteren
- **Loss Functie testen:** Probeer Huber loss voor potentieel stabielere training
- **Epsilon decay aanpassen:** Afhankelijk van de snelheid waarmee je agent leert, kan een snellere of langzamere epsilon decay beter werken
- **Netwerkarchitectuur:** Experimenteren met aantal lagen, units, en dropout kan nuttig zijn
- **Early stopping en checkpoints:** Handig voor lange trainingssessies

---

## Samenvatting

Deze agent implementeert een geavanceerde DQN met:
- Double DQN updates
- Batch Normalization en Dropout
- Epsilon-greedy exploratie met decay
- Replay buffer met potentie voor prioritized replay
- Opslaan en laden van model en trainingsstatus

De code is robuust opgezet en kan verder worden uitgebreid en getuned afhankelijk van jouw behoeften.

Heb je vragen of wil je hulp bij uitbreidingen? Laat het weten!

---
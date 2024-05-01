import parametres_entrainement as pe

def main(HAPS, n, nb_steps):
    score_bete = []
    for _ in range(nb_steps):
        HAPS.next_state([0 for k in range(n)])
        score_bete.append(HAPS.get_reward())

    HAPS.plot(title = f"passif : distance moyenne {sum(score_bete)/nb_steps}, distance finale {score_bete[-1]}")
# -*- encoding=utf-8 -*-

import GmmVSG as Vsg


if __name__ == "__main__":
    gmm_generator = Vsg.GmmVSG(seed=60)
    gmm_generator.fit("../data/23items.csv", 2, 7, [i for i in range(1, 24, 1)], 'akronAbrasion')
    new_df_data = gmm_generator.samples(200)
    print(new_df_data)

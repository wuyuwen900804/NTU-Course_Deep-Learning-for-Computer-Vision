from main import *
import matplotlib.pyplot as plt

def output_eval_img():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_T = 400  # 500
    n_classes = 20
    n_feat = 128  # 128 ok, 256 better (but slower)

    mnistm_save_dir = f"part1/Output_folder/mnistm/"
    svhn_save_dir = f"part1/Output_folder/svhn/"
    pt_dir = f"part1/output/model_99.pth"

    ddpm = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes),
                betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.load_state_dict(torch.load(pt_dir, map_location="cuda"))
    ddpm.to(device)

    ddpm.eval()
    with torch.no_grad():
        n_sample = 50 * n_classes
        x_gen, x_store = ddpm.sample(n_sample, (3, 28, 28), device, guide_w=2)
        mnistm_count, svhn_count = 0, 0
        for i in range(n_sample):
            if i%20 <= 9:
                save_image(x_gen[i], mnistm_save_dir + f"{i%10}_{int((mnistm_count-mnistm_count%10)/10)+1:03d}.png")
                mnistm_count += 1
            else:
                save_image(x_gen[i], svhn_save_dir + f"{i%10}_{int((svhn_count-svhn_count%10)/10)+1:03d}.png")
                svhn_count += 1


def output_report_img():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_T = 400
    n_classes = 20
    n_feat = 128

    save_dir = f"part1/report/"
    pt_dir = f"part1/output/model_99.pth"

    ddpm = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes),
                betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.load_state_dict(torch.load(pt_dir, map_location="cuda"))
    ddpm.to(device)

    ddpm.eval()
    with torch.no_grad():
        n_sample = 10 * n_classes
        x_gen, x_store = ddpm.sample(n_sample, (3, 28, 28), device, guide_w=2)

        mnistm_x_concat_list, svhn_x_concat_list = [], []
        mnistm_x_column_list, svhn_x_column_list = [], []
        for i in range(n_sample):
            if i % n_classes <= 9:
                mnistm_x_column_list.append(x_gen[i])
                if i % 10 == 9:
                    mnistm_column = torch.cat(mnistm_x_column_list, dim=1)
                    mnistm_x_concat_list.append(mnistm_column)
                    mnistm_x_column_list = []
            else:
                svhn_x_column_list.append(x_gen[i])
                if i % 10 == 9:
                    svhn_column = torch.cat(svhn_x_column_list, dim=1)
                    svhn_x_concat_list.append(svhn_column)
                    svhn_x_column_list = []
        mnistm_x_concat = torch.cat(mnistm_x_concat_list, dim=2)
        svhn_x_concat = torch.cat(svhn_x_concat_list, dim=2)
        save_image(mnistm_x_concat, save_dir + "mnistm_concat_img.png")
        save_image(svhn_x_concat, save_dir + "svhn_concat_img.png")
        
        timesteps, n_sample, channels, height, width = x_store.shape
        for t in range(timesteps):
            # mnistm
            img = x_store[t, 0]  # first sample
            img = np.transpose(img, (1, 2, 0))
            img = (img - img.min()) / (img.max() - img.min())
            plt.imsave(os.path.join(save_dir, f"mnistm_timestep_{t}.png"), img)
            # svhn
            img = x_store[t, 10]  # first sample
            img = np.transpose(img, (1, 2, 0))
            img = (img - img.min()) / (img.max() - img.min())
            plt.imsave(os.path.join(save_dir, f"svhn_timestep_{t}.png"), img)
            
if __name__ == "__main__":
    # output_eval_img()
    output_report_img()
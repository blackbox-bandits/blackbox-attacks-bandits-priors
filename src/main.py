import torch as ch
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.nn.modules import Upsample
import argparse
import json

ch.set_default_tensor_type('torch.cuda.FloatTensor')
IMAGENET_SL = 299
IMAGENET_PATH="PATH_TO_IMAGENET_VALIDATION_SET"

model_to_fool = models.inception_v3(pretrained=True).cuda()
model_to_fool = DataParallel(model_to_fool)

model_to_fool.eval()
imagenet = ImageFolder(IMAGENET_PATH, 
                        transforms.Compose([
                            transforms.Resize(IMAGENET_SL),
                            transforms.CenterCrop(IMAGENET_SL),
                            transforms.ToTensor()
                        ]))

def norm(t):
    """
    Takes the norm, treating an n-dimensional tensor as a batch of vectors:
    If x has shape (a, b, c, d), we flatten b, c, d, return the norm along axis 1.
    """
    norm_vec = ch.sqrt(t.pow(2).sum(dim=[1,2,3])).view(-1, 1, 1, 1)
    norm_vec += (norm_vec == 0).float()*1e-8
    return norm_vec

###
# Different optimization steps
# All take the form of func(x, g, lr)
# eg: exponentiated gradients
# l2/linf: projected gradient descent
###

def eg_step(x, g, lr):
    real_x = (x + 1)/2 # from [-1, 1] to [0, 1]
    pos = real_x*ch.exp(lr*g)
    neg = (1-real_x)*ch.exp(-lr*g)
    new_x = pos/(pos+neg)
    return new_x*2-1

def linf_step(x, g, lr):
    return x + lr*ch.sign(g)

def gd_prior_step(x, g, lr):
    return x + lr*g
   
def l2_image_step(x, g, lr):
    return x + lr*g/norm(g)

##
# Projection steps for l2 and linf constraints:
# All take the form of func(new_x, old_x, epsilon)
##

def l2_proj(image, eps):
    orig = image.clone()
    def proj(new_x):
        delta = new_x - orig
        out_of_bounds_mask = (norm(delta) > eps).float()
        x = (orig + eps*delta/norm(delta))*out_of_bounds_mask
        x += new_x*(1-out_of_bounds_mask)
        return x
    return proj

def linf_proj(image, eps):
    orig = image.clone()
    def proj(new_x):
        return orig + ch.clamp(new_x - orig, -eps, eps)
    return proj

##
# Main functions
##

def make_adversarial_examples(image, true_label, args):
    '''
    The main process for generating adversarial examples with priors.
    '''
    # Initial setup
    prior_size = IMAGENET_SL if not args.tiling else args.tile_size
    upsampler = Upsample(size=(IMAGENET_SL, IMAGENET_SL))
    total_queries = ch.zeros(args.batch_size)
    prior = ch.zeros(args.batch_size, 3, prior_size, prior_size)
    dim = prior.nelement()/args.batch_size
    prior_step = gd_prior_step if args.mode == 'l2' else eg_step
    image_step = l2_image_step if args.mode == 'l2' else linf_step
    proj_maker = l2_proj if args.mode == 'l2' else linf_proj
    proj_step = proj_maker(image, args.epsilon)

    # Loss function
    criterion = ch.nn.CrossEntropyLoss(reduction='none')
    L =  lambda x: criterion(model_to_fool(x), true_label)
    losses = L(image)

    # Original classifications
    orig_images = image.clone()
    orig_classes = model_to_fool(image).argmax(1).cuda()
    correct_classified_mask = (orig_classes == true_label).float()
    total_ims = correct_classified_mask.sum()
    not_dones_mask = correct_classified_mask.clone()

    while not ch.any(total_queries > args.max_queries):
        if not args.nes:
            ## Updating the prior: 
            # Create noise for exporation, estimate the gradient, and take a PGD step
            exp_noise = args.exploration*ch.randn_like(prior)/(dim**0.5) 
            # Query deltas for finite difference estimator
            q1 = upsampler(prior + exp_noise)
            q2 = upsampler(prior - exp_noise)
            # Loss points for finite difference estimator
            l1 = L(image + args.fd_eta*q1/norm(q1)) # L(prior + c*noise)
            l2 = L(image + args.fd_eta*q2/norm(q2)) # L(prior - c*noise)
            # Finite differences estimate of directional derivative
            est_deriv = (l1 - l2)/(args.fd_eta*args.exploration)
            # 2-query gradient estimate
            est_grad = est_deriv.view(-1, 1, 1, 1)*exp_noise
            # Update the prior with the estimated gradient
            prior = prior_step(prior, est_grad, args.online_lr)
        else:
            prior = ch.zeros_like(image)
            for _ in range(args.gradient_iters):
                exp_noise = ch.randn_like(image)/(dim**0.5) 
                est_deriv = (L(image + args.fd_eta*exp_noise) - L(image - args.fd_eta*exp_noise))/args.fd_eta
                prior += est_deriv.view(-1, 1, 1, 1)*exp_noise

        # Preserve images that are already done
        prior = prior*not_dones_mask.view(-1, 1, 1, 1)

        ## Update the image:
        # take a pgd step using the prior
        new_im = image_step(image, upsampler(prior), args.image_lr)
        image = proj_step(new_im)
        image = ch.clamp(image, 0, 1)
        if args.mode == 'l2':
            if not ch.all(norm(image - orig_images) <= args.epsilon + 1e-3):
                raise ValueError("OOB")
        else:
            if not (image - orig_images).max() <= args.epsilon + 1e-3:
                raise ValueError("OOB")

        ## Continue query count
        not_dones_mask = not_dones_mask*((model_to_fool(image).argmax(1) == true_label).float())
        total_queries += 2*args.gradient_iters*not_dones_mask

        ## Logging stuff
        new_losses = L(image)
        success_mask = (1 - not_dones_mask)*correct_classified_mask
        num_success = success_mask.sum()
        current_success_rate = (num_success/correct_classified_mask.sum()).cpu().item()
        success_queries = ((success_mask*total_queries).sum()/num_success).cpu().item()
        not_done_loss = ((new_losses*not_dones_mask).sum()/not_dones_mask.sum()).cpu().item()
        max_curr_queries = total_queries.max().cpu().item()
        if args.log_progress:
            print("Queries: %d | Success rate: %f | Average queries: %f" % (max_curr_queries, current_success_rate, success_queries))
        if current_success_rate == 1.0:
            break

    # Return results
    return {
            'average_queries': success_queries, # Average queries for this batch
            'num_correctly_classified': correct_classified_mask.sum().cpu().item(), # Number of originally correctly classified images
            'success_rate': current_success_rate, # Success rate
            'images_orig': orig_images.cpu().numpy(), # Original images
            'images_adv': image.cpu().numpy(), # Adversarial images
            'all_queries': total_queries.cpu().numpy(), # Number of queries used for each image
            'correctly_classified': correct_classified_mask.cpu().numpy(), # 0/1 mask for whether image was originally classified
            'success': success_mask.cpu().numpy() # 0/1 mask for whether the attack succeeds on each image
    }

def main(args):
    imagenet_loader = DataLoader(imagenet, batch_size=args.batch_size)
    average_queries_per_success = 0.0
    total_correctly_classified_ims = 0.0
    success_rate_total = 0.0
    num_batches = 0
    for i, (images, targets) in enumerate(imagenet_loader):
        if i*args.batch_size >= 10:
            return average_queries_per_success/total_correctly_classified_ims, \
                    success_rate_total/total_correctly_classified_ims
        res = make_adversarial_examples(images.cuda(), targets.cuda(), args)
        # The results can be analyzed here!
        average_queries_per_success += res['success_rate']*res['average_queries']*res['num_correctly_classified']
        success_rate_total += res['success_rate']*res['num_correctly_classified']
        total_correctly_classified_ims += res['num_correctly_classified']
        num_batches += 1

class Parameters():
    '''
    Parameters class, just a nice way of accessing a dictionary
    > ps = Parameters({"a": 1, "b": 3})
    > ps.A # returns 1
    > ps.B # returns 3
    '''
    def __init__(self, params):
        self.params = params
    
    def __getattr__(self, x):
        return self.params[x.lower()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-queries', type=int)
    parser.add_argument('--fd-eta', type=float, help='\eta, used to estimate the derivative via finite differences')
    parser.add_argument('--image-lr', type=float, help='Learning rate for the image (iterative attack)')
    parser.add_argument('--online-lr', type=float, help='Learning rate for the prior')
    parser.add_argument('--mode', type=str, help='Which lp constraint to run bandits [linf|l2]')
    parser.add_argument('--exploration', type=float, help='\delta, parameterizes the exploration to be done around the prior')
    parser.add_argument('--tile-size', type=int, help='the side length of each tile (for the tiling prior)')
    parser.add_argument('--json-config', type=str, help='a config file to be passed in instead of arguments')
    parser.add_argument('--epsilon', type=float, help='the lp perturbation bound')
    parser.add_argument('--batch-size', type=int, help='batch size for bandits')
    parser.add_argument('--log-progress', action='store_true')
    parser.add_argument('--nes', action='store_true')
    parser.add_argument('--tiling', action='store_true')
    parser.add_argument('--gradient-iters', type=int)
    args = parser.parse_args()

    args_dict = None
    if not args.json_config:
        # If there is no json file, all of the args must be given
        if not args.nes:
            assert not any([x is None for x in [args.fd_eta, args.max_queries, args.image_lr, \
                            args.mode, args.exploration, args.batch_size, args.epsilon]])
        args_dict = vars(args)
    else:
        # If a json file is given, use the JSON file as the base, and then update it with args
        defaults = json.load(open(args.json_config))
        arg_vars = vars(args)
        arg_vars = {k: arg_vars[k] for k in arg_vars if arg_vars[k] is not None}
        defaults.update(arg_vars)
        args = Parameters(defaults)
        args_dict = defaults

    with ch.no_grad():
        print("Queries, Success = ", main(args))


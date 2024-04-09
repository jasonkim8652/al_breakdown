#!/opt/python/3.7.10/bin/python3
import os
import sys
#sys.path.insert(0, '%s/lib'%os.environ['GALAXY_PIPE_HOME'])
import Galaxy
import warnings

warnings.filterwarnings("error")

def main():
    opt = Galaxy.core.ArgumentParser(method='LigPrep',\
            description='''Add hydrogen&charge to a ligand''')
    #
    opt.add_argument('-l', '--lig', dest='lig_fn', metavar='LIG', required=True,\
            help='Ligand structure file')
    opt.add_argument('--method', dest='method', metavar='METHOD', required=False,\
                     default='chimera', help='prep method(babel or chimera)')
    opt.add_argument('--addH', dest='addH', action='store_true', default=False, \
                        help='add hydrogen to ligand')
    opt.add_argument('--delH', dest='delH', action='store_true', default=False, \
                        help='delete hydrogen to ligand')
    opt.add_argument('--skip_charge', dest='add_charge', action='store_false', default=True, \
                        help='skip charge assign')
    if len(sys.argv) == 1:
        opt.print_help()
        return

    fn = opt.parse_args()
    if fn.title == None:
        title = Galaxy.core.fn.name(fn.lig_fn)
    else:
        title = fn.title
    job = Galaxy.initialize(title, mkdir=False)
    #
    lig_fn = Galaxy.core.FilePath(fn.lig_fn)
    prefix = Galaxy.core.fn.prefix(lig_fn)
    output_prefix = '%s_prep'%prefix
    if fn.method=='chimera':
        Galaxy.tools.chimera.convert(job, lig_fn, \
                                    lig_name=output_prefix, \
                                    re_run = True,\
                                    delete_hydrogen=fn.delH, add_hydrogen=fn.addH)
                                    #delete_H=fn.delH, add_H=fn.addH, charge_method='AM1BCC')
    elif fn.method=='babel':
        Galaxy.tools.babel.run(job, lig_fn, \
                                    delete_H=fn.delH, add_H=fn.addH)

if __name__=='__main__':
    main()
    

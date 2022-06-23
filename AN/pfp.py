# %%
import os
import sys

# %%


def setup_pfp_calculator(
    ver="0102",
    # ver = "0012"
    D_flag=True,
    U_flag=True,
    gpu=0,
    return_dummy=True,
    return_instances=True,
    pfpdir='/cafe05/share/MI/PFN',
):
    # --> PFP settings.
    if not os.path.exists(pfpdir):
        try:
            from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
            from pfp_api_client.pfp.estimator import Estimator
            if not return_instances:
                return ASECalculator, Estimator, None
            estimator = Estimator()
            calculator = ASECalculator(estimator)
            return calculator, estimator, None
        
        except Exception as p:
       
            print(p)
            if return_dummy:
                try:
                    from asap3 import EMT
                except ModuleNotFoundError as p:
                    from ase.calculators.emt import EMT
                if not return_instances:
                    return EMT, None, None
                return EMT(), None, None

        #    print("<VER: {}>   <DFT-U: {}>   <D3: {}>".format(ver, U_flag, D_flag))
    if ver == "0102":
        U_flag = True
        sys.path.insert(0, f"{pfpdir}/pfp010202/deepmi1604crystal/")  #
    elif ver == "0012":
        U_flag = True
        sys.path.insert(
            0, f"{pfpdir}/pfp001201/deepmi1604/"
        )  # 従来計算ではver11を使用、energy-shiftなし
    else:
        ver = "0101"
        if U_flag:
            # sys.path.insert (0, "/cafe05/share/MI/PFN/pfp010100/deepmi1604crystal/") # Uパラメータ有、energy-shiftあり
            sys.path.insert(
                0, f"{pfpdir}/pfp010100/deepmi1604/"
            )  # Uパラメータ有、energy-shiftなし
        else:
            U_flag = False
            sys.path.insert(
                0, f"{pfpdir}/pfp010100/deepmi1604wou/"
            )  # Uパラメータ無
    print(f"<VER: {ver}>   <DFT-U: {U_flag}>   <D3: {D_flag}>")
    sys.stdout.flush()
    if not 'ASECalculator' in sys.modules:
        from pfp.calculators.ase_calculator import ASECalculator
        from pfp.nn.models.crystal import model_builder
        from pfp.nn.estimator_base import EstimatorCalcMode

    if not return_instances:
        return ASECalculator, EstimatorCalcMode, model_builder

    estimator = model_builder.build_estimator(gpu)

    if ver == "0101":
        if U_flag is True:
            estimator.set_calc_mode(EstimatorCalcMode.CRYSTAL)
        else:
            estimator.set_calc_mode(EstimatorCalcMode.CRYSTAL_U0)

    calculator = ASECalculator(estimator)
    print(f"{estimator.calc_mode}")

    if D_flag:
        # from ase.calculators.dftd3 import DFTD3
        # d3 = DFTD3 (dft = calculator, xc = "pbe")
        from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

        calculator = TorchDFTD3Calculator(dft=calculator, device="cuda:0", damping="bj")
        d3 = TorchDFTD3Calculator (dft = calculator, device = "cuda:0", damping = "bj")
        print("D3 correction: {}".format(calculator))
    
    return calculator, estimator, model_builder


# 関数群




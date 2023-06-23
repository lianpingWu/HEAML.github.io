# Initialization
clear
units               metal
dimension           3
boundary            p p p
atom_style          atomic
read_data           gblmp

# parameter setting 
variable            temperature equal 270
variable            dt equal 0.001
variable            T_P equal 10.0*dt
variable            P_P equal 500.0*dt
variable            step equal ntimestep
variable            Pxx equal "-pxx"
variable            Pyy equal "-pyy"
variable            Pzz equal "-pzz"
variable            Pxy equal "-pxy"
variable            Pxz equal "-pxz"
variable            Pyz equal "-pyz"

# Force setting
pair_style          PointFF 6.0 45
pair_coeff          * * jit_epoch_37.pth 1.66 1.83 1.88 1.91 2.20

#compute
compute pstress all stress/atom NULL
compute p all reduce sum c_pstress[1] c_pstress[2] c_pstress[3]
variable pressure equal -(c_p[1]+c_p[2]+c_p[3])/(3.0*vol)

compute csym all centro/atom fcc
compute poten all pe/atom
compute penew all reduce sum c_poten

# dynamic setting
reset_timestep      0
timestep            ${dt}
thermo 100
thermo_style custom step temp press pe etotal

#------------------relax ---------------------
velocity all create ${temperature} 726381 mom yes rot no
fix 1 all npt temp ${temperature} ${temperature} ${T_P} iso 0.0 0.0 ${P_P} drag 1.0
fix 2 all print 100 "$(step) $(temp) $(vol) $(pe) ${Pxx} ${Pyy} ${Pzz} ${Pxy} ${Pxz} ${Pyz}" file ./save/relax_record.dat screen no

dump 1 all custom 1000 ./dump/dump.* id type x y z

run 5000
unfix 1

#MC
fix 1 all npt temp ${temperature} ${temperature} ${T_P} iso 0.0 0.0 ${P_P} drag 1.0

label loop_MC
variable i loop 30

fix 5 all atom/swap 100 100 12345 ${temperature} types 1 2 ke no semi-grand no
run 5000
unfix 5

fix 5 all atom/swap 100 100 12345 ${temperature} types 2 3 ke no semi-grand no
run 5000
unfix 5

fix 5 all atom/swap 100 100 12345 ${temperature} types 3 4 ke no semi-grand no
run 5000
unfix 5

fix 5 all atom/swap 100 100 12345 ${temperature} types 1 3 ke no semi-grand no
run 5000
unfix 5

fix 5 all atom/swap 100 100 12345 ${temperature} types 2 4 ke no semi-grand no
run 5000
unfix 5

fix 5 all atom/swap 100 100 12345 ${temperature} types 4 5 ke no semi-grand no
run 5000
unfix 5

fix 5 all atom/swap 100 100 12345 ${temperature} types 1 4 ke no semi-grand no
run 5000
unfix 5

fix 5 all atom/swap 100 100 12345 ${temperature} types 2 5 ke no semi-grand no
run 5000
unfix 5

fix 5 all atom/swap 100 100 12345 ${temperature} types 3 5 ke no semi-grand no
run 5000
unfix 5

fix 5 all atom/swap 100 100 12345 ${temperature} types 5 1 ke no semi-grand no
run 5000
unfix 5

next i
jump in.MC loop_MC

fix f_npt all npt temp 200 200 0.1 aniso 0.0 0.0 0.1
run 5000


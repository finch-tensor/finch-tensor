Compiling MLIR code:
module {
  func.func @scansearch(
    %arr: memref<?xindex>, %x: index, %lo: index, %hi: index
  ) -> index attributes {llvm.emit_c_interface} {
    %1 = arith.constant 1 : index
    %g:2 = scf.while (%d = %1, %p = %lo) : (index, index) -> (index, index) {
      %plt = arith.cmpi slt, %p, %hi : index
      %cond = scf.if %plt -> (i1) {
        %ap = memref.load %arr[%p] : memref<?xindex>
        %al = arith.cmpi slt, %ap, %x : index
        scf.yield %al : i1
      } else {
        %f = arith.constant false
        scf.yield %f : i1
      }
      scf.condition(%cond) %d, %p : index, index
    } do {
    ^bb0(%d: index, %p: index):
      %d2 = arith.shli %d, %1 : index
      %p2 = arith.addi %p, %d2 : index
      scf.yield %d2, %p2 : index, index
    }
    %lo1 = arith.subi %g#1, %g#0 : index
    %minp = arith.minsi %g#1, %hi : index
    %hi1 = arith.addi %minp, %1 : index
    %b:2 = scf.while (%l = %lo1, %h = %hi1) : (index, index) -> (index, index) {
      %hm1 = arith.subi %h, %1 : index
      %go = arith.cmpi slt, %l, %hm1 : index
      scf.condition(%go) %l, %h : index, index
    } do {
    ^bb0(%l: index, %h: index):
      %diff = arith.subi %h, %l : index
      %half = arith.shrsi %diff, %1 : index
      %m = arith.addi %l, %half : index
      %am = memref.load %arr[%m] : memref<?xindex>
      %al = arith.cmpi slt, %am, %x : index
      %l2, %h2 = scf.if %al -> (index, index) {
        scf.yield %m, %h : index, index
      } else {
        scf.yield %l, %m : index, index
      }
      scf.yield %l2, %h2 : index, index
    }
    return %b#1 : index
  }

  func.func @main(%_A_15: !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>, %_A_16: !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>, %_A_7: !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>, %_ret: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %v = llvm.extractvalue %_A_15[0, 0, 0, 0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_2 = builtin.unrealized_conversion_cast %v : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_3 = llvm.extractvalue %_A_15[0, 0, 3] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_4 = builtin.unrealized_conversion_cast %v_3 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_5 = llvm.extractvalue %_A_15[0, 0, 4] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_6 = builtin.unrealized_conversion_cast %v_5 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_7 = llvm.extractvalue %_A_15[0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_8 = arith.index_cast %v_7 : i64 to index
    %v_9 = llvm.extractvalue %_A_15[0, 0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_10 = arith.index_cast %v_9 : i64 to index
    %v_11 = llvm.extractvalue %_A_16[0, 0, 0, 0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_12 = builtin.unrealized_conversion_cast %v_11 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_13 = llvm.extractvalue %_A_16[0, 0, 3] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_14 = builtin.unrealized_conversion_cast %v_13 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_15 = llvm.extractvalue %_A_16[0, 0, 4] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_16 = builtin.unrealized_conversion_cast %v_15 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_17 = llvm.extractvalue %_A_16[0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_18 = arith.index_cast %v_17 : i64 to index
    %v_19 = llvm.extractvalue %_A_16[0, 0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_20 = arith.index_cast %v_19 : i64 to index
    %v_21 = llvm.extractvalue %_A_7[0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_22 = builtin.unrealized_conversion_cast %v_21 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_23 = llvm.extractvalue %_A_7[1, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_24 = arith.index_cast %v_23 : i64 to index
    %v_25 = llvm.extractvalue %_A_7[1, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_26 = arith.index_cast %v_25 : i64 to index
    %v_27 = arith.constant 0 : index
    %v_28 = memref.dim %v_22, %v_27 : memref<?xf64>
    %v_29 = arith.constant 1 : index
    scf.for %v_30 = %v_27 to %v_28 step %v_29 {
      %v_31 = arith.constant 0.0 : f64
      memref.store %v_31, %v_22[%v_30] : memref<?xf64>
    }
    scf.for %v_32 = %v_27 to %v_8 step %v_29 {
      %v_33 = llvm.extractvalue %_A_7[2, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
      %v_34 = arith.index_cast %v_33 : i64 to index
      %v_35 = arith.muli %v_34, %v_32 : index
      %v_36 = llvm.extractvalue %_A_15[0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
      %v_37 = arith.index_cast %v_36 : i64 to index
      %v_38 = arith.muli %v_37, %v_32 : index
      %v_39 = memref.load %v_4[%v_38] : memref<?xindex>
      %v_40 = arith.addi %v_29, %v_38 : index
      %v_41 = memref.load %v_4[%v_40] : memref<?xindex>
      %v_42 = arith.cmpi slt, %v_39, %v_41 : index
      %v_46, %v_47 = scf.if %v_42 -> (index, index) {
        %v_43 = memref.load %v_6[%v_39] : memref<?xindex>
        %v_44 = arith.subi %v_41, %v_29 : index
        %v_45 = memref.load %v_6[%v_44] : memref<?xindex>
        scf.yield %v_45, %v_43 : index, index
      } else {
        scf.yield %v_27, %v_29 : index, index
      }
      %v_48 = memref.load %v_6[%v_39] : memref<?xindex>
      %v_49 = arith.cmpi slt, %v_48, %v_27 : index
      %v_52 = scf.if %v_49 -> (index) {
        %v_50 = arith.subi %v_41, %v_29 : index
        %v_51 = func.call @scansearch(%v_6, %v_27, %v_39, %v_50) : (memref<?xindex>, index, index, index) -> index
        scf.yield %v_51 : index
      } else {
        scf.yield %v_39 : index
      }
      %v_136:3 = scf.while (%v_53 = %v_27, %v_54 = %v_52, %v_55 = %v_47) : (index, index, index) -> (index, index, index) {
        %v_56 = arith.addi %v_29, %v_46 : index
        %v_57 = arith.minsi %v_10, %v_56 : index
        %v_58 = arith.cmpi slt, %v_55, %v_57 : index
        scf.condition(%v_58) %v_53, %v_54, %v_55 : index, index, index
      } do {
        ^bb_2(%v_53: index, %v_54: index, %v_55: index):
        scf.for %v_59 = %v_27 to %v_20 step %v_29 {
          %v_60 = llvm.extractvalue %_A_7[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
          %v_61 = arith.index_cast %v_60 : i64 to index
          %v_62 = arith.muli %v_61, %v_59 : index
          %v_63 = arith.addi %v_35, %v_62 : index
          %v_64 = arith.constant 0.0 : f64
          %v_65 = memref.load %v_22[%v_63] : memref<?xf64>
          %v_66 = arith.addf %v_64, %v_65 : f64
          memref.store %v_66, %v_22[%v_63] : memref<?xf64>
        }
        %v_67 = arith.maxsi %v_53, %v_55 : index
        %v_68 = arith.addi %v_29, %v_55 : index
        scf.for %v_69 = %v_67 to %v_68 step %v_29 {
          %v_70 = llvm.extractvalue %_A_16[0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
          %v_71 = arith.index_cast %v_70 : i64 to index
          %v_72 = arith.muli %v_71, %v_69 : index
          %v_73 = memref.load %v_14[%v_72] : memref<?xindex>
          %v_74 = arith.addi %v_29, %v_72 : index
          %v_75 = memref.load %v_14[%v_74] : memref<?xindex>
          %v_76 = arith.cmpi slt, %v_73, %v_75 : index
          %v_80, %v_81 = scf.if %v_76 -> (index, index) {
            %v_77 = memref.load %v_16[%v_73] : memref<?xindex>
            %v_78 = arith.subi %v_75, %v_29 : index
            %v_79 = memref.load %v_16[%v_78] : memref<?xindex>
            scf.yield %v_79, %v_77 : index, index
          } else {
            scf.yield %v_27, %v_29 : index, index
          }
          %v_82 = memref.load %v_16[%v_73] : memref<?xindex>
          %v_83 = arith.cmpi slt, %v_82, %v_27 : index
          %v_86 = scf.if %v_83 -> (index) {
            %v_84 = arith.subi %v_75, %v_29 : index
            %v_85 = func.call @scansearch(%v_16, %v_27, %v_73, %v_84) : (memref<?xindex>, index, index, index) -> index
            scf.yield %v_85 : index
          } else {
            scf.yield %v_73 : index
          }
          %v_120:3 = scf.while (%v_87 = %v_27, %v_88 = %v_86, %v_89 = %v_81) : (index, index, index) -> (index, index, index) {
            %v_90 = arith.addi %v_29, %v_80 : index
            %v_91 = arith.minsi %v_20, %v_90 : index
            %v_92 = arith.cmpi slt, %v_89, %v_91 : index
            scf.condition(%v_92) %v_87, %v_88, %v_89 : index, index, index
          } do {
            ^bb(%v_87: index, %v_88: index, %v_89: index):
            %v_93 = arith.addi %v_29, %v_89 : index
            %v_94 = arith.minsi %v_93, %v_89 : index
            scf.for %v_95 = %v_87 to %v_94 step %v_29 {
              %v_96 = llvm.extractvalue %_A_7[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
              %v_97 = arith.index_cast %v_96 : i64 to index
              %v_98 = arith.muli %v_97, %v_95 : index
              %v_99 = arith.addi %v_35, %v_98 : index
              %v_100 = arith.constant 0.0 : f64
              %v_101 = memref.load %v_22[%v_99] : memref<?xf64>
              %v_102 = arith.addf %v_100, %v_101 : f64
              memref.store %v_102, %v_22[%v_99] : memref<?xf64>
            }
            %v_103 = arith.maxsi %v_87, %v_89 : index
            %v_104 = arith.addi %v_29, %v_89 : index
            scf.for %v_105 = %v_103 to %v_104 step %v_29 {
              %v_106 = llvm.extractvalue %_A_7[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
              %v_107 = arith.index_cast %v_106 : i64 to index
              %v_108 = arith.muli %v_107, %v_105 : index
              %v_109 = arith.addi %v_35, %v_108 : index
              %v_110 = memref.load %v_22[%v_109] : memref<?xf64>
              %v_111 = memref.load %v_2[%v_54] : memref<?xf64>
              %v_112 = memref.load %v_12[%v_88] : memref<?xf64>
              %v_113 = arith.mulf %v_111, %v_112 : f64
              %v_114 = arith.addf %v_110, %v_113 : f64
              memref.store %v_114, %v_22[%v_109] : memref<?xf64>
            }
            %v_115 = arith.addi %v_29, %v_89 : index
            %v_116 = arith.addi %v_29, %v_88 : index
            %v_117 = arith.cmpi slt, %v_116, %v_75 : index
            %v_119 = scf.if %v_117 -> (index) {
              %v_118 = memref.load %v_16[%v_116] : memref<?xindex>
              scf.yield %v_118 : index
            } else {
              scf.yield %v_20 : index
            }
            scf.yield %v_115, %v_116, %v_119 : index, index, index
          }
          %v_121 = arith.addi %v_29, %v_80 : index
          %v_122 = arith.maxsi %v_27, %v_121 : index
          scf.for %v_123 = %v_122 to %v_20 step %v_29 {
            %v_124 = llvm.extractvalue %_A_7[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
            %v_125 = arith.index_cast %v_124 : i64 to index
            %v_126 = arith.muli %v_125, %v_123 : index
            %v_127 = arith.addi %v_35, %v_126 : index
            %v_128 = arith.constant 0.0 : f64
            %v_129 = memref.load %v_22[%v_127] : memref<?xf64>
            %v_130 = arith.addf %v_128, %v_129 : f64
            memref.store %v_130, %v_22[%v_127] : memref<?xf64>
          }
        }
        %v_131 = arith.addi %v_29, %v_55 : index
        %v_132 = arith.addi %v_29, %v_54 : index
        %v_133 = arith.cmpi slt, %v_132, %v_41 : index
        %v_135 = scf.if %v_133 -> (index) {
          %v_134 = memref.load %v_6[%v_132] : memref<?xindex>
          scf.yield %v_134 : index
        } else {
          scf.yield %v_10 : index
        }
        scf.yield %v_131, %v_132, %v_135 : index, index, index
      }
      scf.for %v_137 = %v_27 to %v_20 step %v_29 {
        %v_138 = llvm.extractvalue %_A_7[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
        %v_139 = arith.index_cast %v_138 : i64 to index
        %v_140 = arith.muli %v_139, %v_137 : index
        %v_141 = arith.addi %v_35, %v_140 : index
        %v_142 = arith.constant 0.0 : f64
        %v_143 = memref.load %v_22[%v_141] : memref<?xf64>
        %v_144 = arith.addf %v_142, %v_143 : f64
        memref.store %v_144, %v_22[%v_141] : memref<?xf64>
      }
    }
    %v_145 = llvm.mlir.undef : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    %v_146 = llvm.insertvalue %_A_7, %v_145[0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    llvm.store %v_146, %_ret : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>, !llvm.ptr
    func.return
  }
}

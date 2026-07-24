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

  func.func @main(%_A_9: !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>, %_A_10: !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>, %_A_4: !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>, %_ret: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %v = llvm.extractvalue %_A_9[0, 0, 0, 0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_2 = builtin.unrealized_conversion_cast %v : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_3 = llvm.extractvalue %_A_9[0, 0, 3] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_4 = builtin.unrealized_conversion_cast %v_3 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_5 = llvm.extractvalue %_A_9[0, 0, 4] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_6 = builtin.unrealized_conversion_cast %v_5 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_7 = llvm.extractvalue %_A_9[0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_8 = arith.index_cast %v_7 : i64 to index
    %v_9 = llvm.extractvalue %_A_9[0, 0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_10 = arith.index_cast %v_9 : i64 to index
    %v_11 = llvm.extractvalue %_A_10[0, 0, 0, 0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_12 = builtin.unrealized_conversion_cast %v_11 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_13 = llvm.extractvalue %_A_10[0, 0, 3] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_14 = builtin.unrealized_conversion_cast %v_13 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_15 = llvm.extractvalue %_A_10[0, 0, 4] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_16 = builtin.unrealized_conversion_cast %v_15 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_17 = llvm.extractvalue %_A_10[0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_18 = arith.index_cast %v_17 : i64 to index
    %v_19 = llvm.extractvalue %_A_10[0, 0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_20 = arith.index_cast %v_19 : i64 to index
    %v_21 = llvm.extractvalue %_A_4[0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_22 = builtin.unrealized_conversion_cast %v_21 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_23 = llvm.extractvalue %_A_4[1, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_24 = arith.index_cast %v_23 : i64 to index
    %v_25 = llvm.extractvalue %_A_4[1, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_26 = arith.index_cast %v_25 : i64 to index
    %v_27 = arith.constant 0 : index
    %v_28 = memref.dim %v_22, %v_27 : memref<?xf64>
    %v_29 = arith.constant 1 : index
    scf.for %v_30 = %v_27 to %v_28 step %v_29 {
      %v_31 = arith.constant 0.0 : f64
      memref.store %v_31, %v_22[%v_30] : memref<?xf64>
    }
    scf.for %v_32 = %v_27 to %v_18 step %v_29 {
      %v_33 = llvm.extractvalue %_A_4[2, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
      %v_34 = arith.index_cast %v_33 : i64 to index
      %v_35 = arith.muli %v_34, %v_32 : index
      %v_36 = arith.addi %v_27, %v_35 : index
      %v_37 = llvm.extractvalue %_A_9[0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
      %v_38 = arith.index_cast %v_37 : i64 to index
      %v_39 = arith.muli %v_38, %v_32 : index
      %v_40 = arith.addi %v_27, %v_39 : index
      %v_41 = llvm.extractvalue %_A_10[0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
      %v_42 = arith.index_cast %v_41 : i64 to index
      %v_43 = arith.muli %v_42, %v_32 : index
      %v_44 = arith.addi %v_27, %v_43 : index
      %v_45 = memref.load %v_4[%v_40] : memref<?xindex>
      %v_46 = arith.addi %v_40, %v_29 : index
      %v_47 = memref.load %v_4[%v_46] : memref<?xindex>
      %v_48 = arith.cmpi slt, %v_45, %v_47 : index
      %v_52, %v_53 = scf.if %v_48 -> (index, index) {
        %v_49 = memref.load %v_6[%v_45] : memref<?xindex>
        %v_50 = arith.subi %v_47, %v_29 : index
        %v_51 = memref.load %v_6[%v_50] : memref<?xindex>
        scf.yield %v_51, %v_49 : index, index
      } else {

        scf.yield %v_27, %v_29 : index, index
      }
      %v_54 = memref.load %v_14[%v_44] : memref<?xindex>
      %v_55 = arith.addi %v_44, %v_29 : index
      %v_56 = memref.load %v_14[%v_55] : memref<?xindex>
      %v_57 = arith.cmpi slt, %v_54, %v_56 : index
      %v_61, %v_62 = scf.if %v_57 -> (index, index) {
        %v_58 = memref.load %v_16[%v_54] : memref<?xindex>
        %v_59 = arith.subi %v_56, %v_29 : index
        %v_60 = memref.load %v_16[%v_59] : memref<?xindex>
        scf.yield %v_60, %v_58 : index, index
      } else {

        scf.yield %v_27, %v_29 : index, index
      }
      %v_63 = memref.load %v_6[%v_45] : memref<?xindex>
      %v_64 = arith.cmpi slt, %v_63, %v_27 : index
      %v_67 = scf.if %v_64 -> (index) {
        %v_65 = arith.subi %v_47, %v_29 : index
        %v_66 = func.call @scansearch(%v_6, %v_27, %v_45, %v_65) : (memref<?xindex>, index, index, index) -> index
        scf.yield %v_66 : index
      } else {
        scf.yield %v_45 : index
      }
      %v_68 = memref.load %v_16[%v_54] : memref<?xindex>
      %v_69 = arith.cmpi slt, %v_68, %v_27 : index
      %v_72 = scf.if %v_69 -> (index) {
        %v_70 = arith.subi %v_56, %v_29 : index
        %v_71 = func.call @scansearch(%v_16, %v_27, %v_54, %v_70) : (memref<?xindex>, index, index, index) -> index
        scf.yield %v_71 : index
      } else {
        scf.yield %v_54 : index
      }
      %v_174:5 = scf.while (%v_73 = %v_27, %v_74 = %v_67, %v_75 = %v_53, %v_76 = %v_72, %v_77 = %v_62) : (index, index, index, index, index) -> (index, index, index, index, index) {
        %v_78 = arith.minsi %v_75, %v_77 : index
        %v_79 = arith.addi %v_52, %v_29 : index
        %v_80 = arith.minsi %v_20, %v_79 : index
        %v_81 = arith.addi %v_61, %v_29 : index
        %v_82 = arith.minsi %v_80, %v_81 : index
        %v_83 = arith.cmpi slt, %v_78, %v_82 : index
        scf.condition(%v_83) %v_73, %v_74, %v_75, %v_76, %v_77 : index, index, index, index, index
      } do {
        ^bb(%v_73: index, %v_74: index, %v_75: index, %v_76: index, %v_77: index):
        %v_84 = arith.minsi %v_75, %v_77 : index
        %v_85 = arith.cmpi eq, %v_84, %v_75 : index
        %v_126, %v_127, %v_128 = scf.if %v_85 -> (index, index, index) {
          %v_86 = arith.addi %v_75, %v_29 : index
          %v_87 = arith.minsi %v_86, %v_75 : index
          %v_88 = arith.minsi %v_87, %v_77 : index
          scf.for %v_89 = %v_73 to %v_88 step %v_29 {
            %v_90 = llvm.extractvalue %_A_4[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
            %v_91 = arith.index_cast %v_90 : i64 to index
            %v_92 = arith.muli %v_91, %v_89 : index
            %v_93 = arith.addi %v_36, %v_92 : index
          }
          %v_94 = arith.maxsi %v_73, %v_77 : index
          %v_95 = arith.addi %v_75, %v_29 : index
          %v_96 = arith.minsi %v_95, %v_75 : index
          scf.for %v_97 = %v_94 to %v_96 step %v_29 {
            %v_98 = llvm.extractvalue %_A_4[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
            %v_99 = arith.index_cast %v_98 : i64 to index
            %v_100 = arith.muli %v_99, %v_97 : index
            %v_101 = arith.addi %v_36, %v_100 : index
          }
          %v_102 = arith.maxsi %v_73, %v_75 : index
          %v_103 = arith.addi %v_75, %v_29 : index
          %v_104 = arith.minsi %v_103, %v_77 : index
          scf.for %v_105 = %v_102 to %v_104 step %v_29 {
            %v_106 = llvm.extractvalue %_A_4[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
            %v_107 = arith.index_cast %v_106 : i64 to index
            %v_108 = arith.muli %v_107, %v_105 : index
            %v_109 = arith.addi %v_36, %v_108 : index
          }
          %v_110 = arith.maxsi %v_73, %v_75 : index
          %v_111 = arith.maxsi %v_110, %v_77 : index
          %v_112 = arith.addi %v_75, %v_29 : index
          scf.for %v_113 = %v_111 to %v_112 step %v_29 {
            %v_114 = llvm.extractvalue %_A_4[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
            %v_115 = arith.index_cast %v_114 : i64 to index
            %v_116 = arith.muli %v_115, %v_113 : index
            %v_117 = arith.addi %v_36, %v_116 : index
            %v_118 = memref.load %v_2[%v_74] : memref<?xf64>
            %v_119 = memref.load %v_12[%v_76] : memref<?xf64>
            %v_120 = arith.mulf %v_118, %v_119 : f64
            memref.store %v_120, %v_22[%v_117] : memref<?xf64>
          }
          %v_121 = arith.addi %v_75, %v_29 : index
          %v_122 = arith.addi %v_74, %v_29 : index
          %v_123 = arith.cmpi slt, %v_122, %v_47 : index
          %v_125 = scf.if %v_123 -> (index) {
            %v_124 = memref.load %v_6[%v_122] : memref<?xindex>
            scf.yield %v_124 : index
          } else {

            scf.yield %v_10 : index
          }
          scf.yield %v_121, %v_122, %v_125 : index, index, index
        } else {
          scf.yield %v_73, %v_74, %v_75 : index, index, index
        }
        %v_129 = arith.minsi %v_128, %v_77 : index
        %v_130 = arith.cmpi eq, %v_129, %v_77 : index
        %v_171, %v_172, %v_173 = scf.if %v_130 -> (index, index, index) {
          %v_131 = arith.addi %v_77, %v_29 : index
          %v_132 = arith.minsi %v_131, %v_128 : index
          %v_133 = arith.minsi %v_132, %v_77 : index
          scf.for %v_134 = %v_126 to %v_133 step %v_29 {
            %v_135 = llvm.extractvalue %_A_4[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
            %v_136 = arith.index_cast %v_135 : i64 to index
            %v_137 = arith.muli %v_136, %v_134 : index
            %v_138 = arith.addi %v_36, %v_137 : index
          }
          %v_139 = arith.maxsi %v_126, %v_77 : index
          %v_140 = arith.addi %v_77, %v_29 : index
          %v_141 = arith.minsi %v_140, %v_128 : index
          scf.for %v_142 = %v_139 to %v_141 step %v_29 {
            %v_143 = llvm.extractvalue %_A_4[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
            %v_144 = arith.index_cast %v_143 : i64 to index
            %v_145 = arith.muli %v_144, %v_142 : index
            %v_146 = arith.addi %v_36, %v_145 : index
          }
          %v_147 = arith.maxsi %v_126, %v_128 : index
          %v_148 = arith.addi %v_77, %v_29 : index
          %v_149 = arith.minsi %v_148, %v_77 : index
          scf.for %v_150 = %v_147 to %v_149 step %v_29 {
            %v_151 = llvm.extractvalue %_A_4[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
            %v_152 = arith.index_cast %v_151 : i64 to index
            %v_153 = arith.muli %v_152, %v_150 : index
            %v_154 = arith.addi %v_36, %v_153 : index
          }
          %v_155 = arith.maxsi %v_126, %v_128 : index
          %v_156 = arith.maxsi %v_155, %v_77 : index
          %v_157 = arith.addi %v_77, %v_29 : index
          scf.for %v_158 = %v_156 to %v_157 step %v_29 {
            %v_159 = llvm.extractvalue %_A_4[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
            %v_160 = arith.index_cast %v_159 : i64 to index
            %v_161 = arith.muli %v_160, %v_158 : index
            %v_162 = arith.addi %v_36, %v_161 : index
            %v_163 = memref.load %v_2[%v_127] : memref<?xf64>
            %v_164 = memref.load %v_12[%v_76] : memref<?xf64>
            %v_165 = arith.mulf %v_163, %v_164 : f64
            memref.store %v_165, %v_22[%v_162] : memref<?xf64>
          }
          %v_166 = arith.addi %v_77, %v_29 : index
          %v_167 = arith.addi %v_76, %v_29 : index
          %v_168 = arith.cmpi slt, %v_167, %v_56 : index
          %v_170 = scf.if %v_168 -> (index) {
            %v_169 = memref.load %v_16[%v_167] : memref<?xindex>
            scf.yield %v_169 : index
          } else {

            scf.yield %v_20 : index
          }
          scf.yield %v_166, %v_167, %v_170 : index, index, index
        } else {
          scf.yield %v_126, %v_76, %v_77 : index, index, index
        }
        scf.yield %v_171, %v_127, %v_128, %v_172, %v_173 : index, index, index, index, index
      }
      %v_175 = arith.addi %v_61, %v_29 : index
      %v_176 = arith.maxsi %v_27, %v_175 : index
      %v_177 = arith.addi %v_52, %v_29 : index
      %v_178 = arith.minsi %v_20, %v_177 : index
      scf.for %v_179 = %v_176 to %v_178 step %v_29 {
        %v_180 = llvm.extractvalue %_A_4[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
        %v_181 = arith.index_cast %v_180 : i64 to index
        %v_182 = arith.muli %v_181, %v_179 : index
        %v_183 = arith.addi %v_36, %v_182 : index
      }
      %v_184 = arith.addi %v_52, %v_29 : index
      %v_185 = arith.maxsi %v_27, %v_184 : index
      %v_186 = arith.addi %v_61, %v_29 : index
      %v_187 = arith.minsi %v_20, %v_186 : index
      scf.for %v_188 = %v_185 to %v_187 step %v_29 {
        %v_189 = llvm.extractvalue %_A_4[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
        %v_190 = arith.index_cast %v_189 : i64 to index
        %v_191 = arith.muli %v_190, %v_188 : index
        %v_192 = arith.addi %v_36, %v_191 : index
      }
      %v_193 = arith.addi %v_52, %v_29 : index
      %v_194 = arith.maxsi %v_27, %v_193 : index
      %v_195 = arith.addi %v_61, %v_29 : index
      %v_196 = arith.maxsi %v_194, %v_195 : index
      scf.for %v_197 = %v_196 to %v_20 step %v_29 {
        %v_198 = llvm.extractvalue %_A_4[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
        %v_199 = arith.index_cast %v_198 : i64 to index
        %v_200 = arith.muli %v_199, %v_197 : index
        %v_201 = arith.addi %v_36, %v_200 : index
      }
    }
    %v_202 = llvm.mlir.undef : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    %v_203 = llvm.insertvalue %_A_4, %v_202[0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    llvm.store %v_203, %_ret : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>, !llvm.ptr
    func.return
  }
}

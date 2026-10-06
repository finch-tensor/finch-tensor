module {
  func.func @sparse_output(%idx: !llvm.ptr, %val: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %v = llvm.getelementptr %idx[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_2 = llvm.load %v : !llvm.ptr -> !llvm.ptr
    %v_3 = llvm.getelementptr %idx[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_4 = llvm.load %v_3 : !llvm.ptr -> i64
    %v_5 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_6 = arith.constant 0 : i64
    %v_7 = arith.constant 1 : i64
    %v_8 = llvm.insertvalue %v_2, %v_5[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_9 = llvm.insertvalue %v_2, %v_8[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_10 = llvm.insertvalue %v_6, %v_9[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_11 = llvm.insertvalue %v_4, %v_10[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_12 = llvm.insertvalue %v_7, %v_11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_13 = builtin.unrealized_conversion_cast %v_12 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_14 = builtin.unrealized_conversion_cast %v_13 : memref<?xindex> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_15 = llvm.alloca %v_7 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_14, %v_15 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_16 = llvm.getelementptr %val[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_17 = llvm.load %v_16 : !llvm.ptr -> !llvm.ptr
    %v_18 = llvm.getelementptr %val[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_19 = llvm.load %v_18 : !llvm.ptr -> i64
    %v_20 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_21 = llvm.insertvalue %v_17, %v_20[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_22 = llvm.insertvalue %v_17, %v_21[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_23 = llvm.insertvalue %v_6, %v_22[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_24 = llvm.insertvalue %v_19, %v_23[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_25 = llvm.insertvalue %v_7, %v_24[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_26 = builtin.unrealized_conversion_cast %v_25 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_27 = builtin.unrealized_conversion_cast %v_26 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_28 = llvm.alloca %v_7 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_27, %v_28 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_29 = arith.constant 1 : index
    %v_30 = arith.index_cast %v_29 : index to i64
    %v_31 = llvm.getelementptr %idx[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_32 = llvm.getelementptr %idx[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_33 = llvm.load %v_32 : !llvm.ptr -> !llvm.ptr
    %v_34 = llvm.call %v_33(%v_31, %v_30) : !llvm.ptr, (!llvm.ptr, i64) -> !llvm.ptr
    %v_35 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_36 = llvm.insertvalue %v_34, %v_35[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_37 = llvm.insertvalue %v_34, %v_36[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_38 = llvm.insertvalue %v_6, %v_37[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_39 = llvm.insertvalue %v_30, %v_38[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_40 = llvm.insertvalue %v_7, %v_39[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_41 = builtin.unrealized_conversion_cast %v_40 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_42 = builtin.unrealized_conversion_cast %v_41 : memref<?xindex> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.store %v_42, %v_15 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_43 = arith.index_cast %v_29 : index to i64
    %v_44 = llvm.getelementptr %val[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_45 = llvm.getelementptr %val[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_46 = llvm.load %v_45 : !llvm.ptr -> !llvm.ptr
    %v_47 = llvm.call %v_46(%v_44, %v_43) : !llvm.ptr, (!llvm.ptr, i64) -> !llvm.ptr
    %v_48 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_49 = llvm.insertvalue %v_47, %v_48[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_50 = llvm.insertvalue %v_47, %v_49[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_51 = llvm.insertvalue %v_6, %v_50[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_52 = llvm.insertvalue %v_43, %v_51[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_53 = llvm.insertvalue %v_7, %v_52[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_54 = builtin.unrealized_conversion_cast %v_53 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_55 = builtin.unrealized_conversion_cast %v_54 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.store %v_55, %v_28 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_56 = llvm.load %v_15 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_57 = builtin.unrealized_conversion_cast %v_56 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_58 = arith.constant 2 : index
    %v_59 = arith.constant 0 : index
    memref.store %v_58, %v_57[%v_59] : memref<?xindex>
    %v_60 = llvm.load %v_28 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_61 = builtin.unrealized_conversion_cast %v_60 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_62 = arith.constant 3.5 : f64
    memref.store %v_62, %v_61[%v_59] : memref<?xf64>
    %v_63 = llvm.load %v_15 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_64 = builtin.unrealized_conversion_cast %v_63 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_65 = llvm.load %v_15 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_66 = builtin.unrealized_conversion_cast %v_65 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_67 = memref.dim %v_66, %v_59 : memref<?xindex>
    %v_68 = arith.index_cast %v_67 : index to i64
    %v_69 = llvm.getelementptr %idx[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_68, %v_69 : i64, !llvm.ptr
    %v_70 = memref.extract_aligned_pointer_as_index %v_64 : memref<?xindex> -> index
    %v_71 = arith.index_cast %v_70 : index to i64
    %v_72 = llvm.inttoptr %v_71 : i64 to !llvm.ptr
    %v_73 = llvm.getelementptr %idx[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_72, %v_73 : !llvm.ptr, !llvm.ptr
    %v_74 = llvm.load %v_28 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_75 = builtin.unrealized_conversion_cast %v_74 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_76 = llvm.load %v_28 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_77 = builtin.unrealized_conversion_cast %v_76 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_78 = memref.dim %v_77, %v_59 : memref<?xf64>
    %v_79 = arith.index_cast %v_78 : index to i64
    %v_80 = llvm.getelementptr %val[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_79, %v_80 : i64, !llvm.ptr
    %v_81 = memref.extract_aligned_pointer_as_index %v_75 : memref<?xf64> -> index
    %v_82 = arith.index_cast %v_81 : index to i64
    %v_83 = llvm.inttoptr %v_82 : i64 to !llvm.ptr
    %v_84 = llvm.getelementptr %val[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_83, %v_84 : !llvm.ptr, !llvm.ptr
    func.return
  }
}
